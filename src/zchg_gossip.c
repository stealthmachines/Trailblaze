/*
 * zchg_gossip.c — Binary gossip protocol for cluster convergence.
 */

#include "zchg_transport.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

static int zchg_gossip_select_peers(const zchg_lattice_t *lattice,
                                    uint32_t *out_peers,
                                    int max_peers) {
    if (!lattice || !out_peers || max_peers <= 0) {
        return 0;
    }

    int out_count = 0;
    for (uint32_t i = 0; i < lattice->peer_count && out_count < max_peers; i++) {
        const zchg_peer_t *peer = &lattice->peers[i];
        if (!peer->is_healthy || peer->ip_addr == 0) {
            continue;
        }
        out_peers[out_count++] = peer->ip_addr;
    }

    return out_count;
}

void zchg_gossip_create_message(const zchg_lattice_t *lattice, zchg_gossip_msg_t *out_msg) {
    if (!lattice || !out_msg) {
        return;
    }

    memset(out_msg, 0, sizeof(*out_msg));
    out_msg->source_ip = lattice->local_ip;
    out_msg->cluster_fingerprint = lattice->cluster_fingerprint;
    out_msg->storage_available = (uint32_t)lattice->my_strands[0].storage_available;

    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        out_msg->strand_weights[i] = (uint8_t)lattice->my_strands[i].authority_weight;
    }
}

int zchg_gossip_broadcast(zchg_transport_server_t *server, const zchg_gossip_msg_t *msg) {
    if (!server || !msg) {
        return -1;
    }

    uint32_t peers[3] = {0};
    int peer_count = zchg_gossip_select_peers(&server->lattice, peers, 3);
    if (peer_count <= 0) {
        return 0;
    }

    for (int i = 0; i < peer_count; i++) {
        if (peers[i] == 0 || peers[i] == server->local_ip) {
            continue;
        }

        zchg_frame_t frame;
        memset(&frame, 0, sizeof(frame));
        frame.header.version = zchg_FRAME_VERSION;
        frame.header.type = zchg_FRAME_GOSSIP;
        frame.header.source_ip = server->local_ip;
        frame.header.payload_len = (uint32_t)sizeof(*msg);
        frame.header.timestamp = (uint64_t)time(NULL) * 1000ULL;
        frame.payload_len = sizeof(*msg);
        frame.payload = (uint8_t *)malloc(sizeof(*msg));
        if (!frame.payload) {
            continue;
        }

        memcpy(frame.payload, msg, sizeof(*msg));
        zchg_frame_t *response = NULL;
        (void)zchg_client_send_frame(server, peers[i], &frame, &response);
        if (response) {
            if (response->payload) {
                free(response->payload);
            }
            free(response);
        }
        free(frame.payload);
    }

    return 0;
}

int zchg_gossip_cycle(zchg_transport_server_t *server) {
    if (!server) {
        return -1;
    }

    zchg_gossip_msg_t msg;
    zchg_gossip_create_message(&server->lattice, &msg);
    return zchg_gossip_broadcast(server, &msg);
}

int zchg_gossip_evict_dead_peers(zchg_lattice_t *lattice) {
    if (!lattice) {
        return -1;
    }

    time_t now = time(NULL);
    int removed = 0;

    for (uint32_t i = 0; i < lattice->peer_count; i++) {
        zchg_peer_t *peer = &lattice->peers[i];
        if (!peer->is_healthy) {
            continue;
        }

        if (peer->last_gossip_in > 0 && (now - peer->last_gossip_in) > (time_t)(zchg_GOSSIP_INTERVAL * 4)) {
            peer->is_healthy = 0;
            removed++;
        }
    }

    return removed;
}
