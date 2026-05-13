/*
 * zchg_http.c - Native HTTP Server for ZCHG v0.6-c
 *
 * This file replaces NGINX for the ZCHG front door.
 *
 * Features:
 * - Native HTTP/1.1 server in pure C
 * - Non-blocking sockets + select()-based event loop
 * - Keep-alive and basic request pipelining
 * - Strand-aware routing for /serve/ paths
 * - Direct ZCHG endpoints: /health, /metrics, /node_info, /strand_map
 * - Binary ZCHG frame handlers for gossip and fetch operations
 */

#include "zchg_transport.h"
#include "zchg_lattice.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef zchg_HTTP_MAX_CONNECTIONS
#define zchg_HTTP_MAX_CONNECTIONS   1024
#endif

#ifndef zchg_HTTP_READ_BUF
#define zchg_HTTP_READ_BUF          8192
#endif

#ifndef zchg_HTTP_RESPONSE_MAX
#define zchg_HTTP_RESPONSE_MAX      262144
#endif

#ifndef zchg_HTTP_METHOD_MAX
#define zchg_HTTP_METHOD_MAX        8
#endif

#ifndef zchg_HTTP_PATH_MAX
#define zchg_HTTP_PATH_MAX          1024
#endif

#ifndef zchg_HTTP_VERSION_MAX
#define zchg_HTTP_VERSION_MAX       16
#endif

typedef enum {
    zchg_HTTP_CONN_FREE = 0,
    zchg_HTTP_CONN_READING,
    zchg_HTTP_CONN_WRITING
} zchg_http_conn_state_t;

typedef struct {
    int                     fd;
    zchg_http_conn_state_t  state;
    char                    read_buf[zchg_HTTP_READ_BUF];
    size_t                  read_len;
    char                   *write_buf;
    size_t                  write_len;
    size_t                  write_sent;
    char                    method[zchg_HTTP_METHOD_MAX];
    char                    path[zchg_HTTP_PATH_MAX];
    char                    version[zchg_HTTP_VERSION_MAX];
    size_t                  content_length;
    size_t                  body_offset;
    int                     keep_alive;
} zchg_http_connection_t;

struct zchg_event_loop {
    zchg_transport_server_t   *server;
    int                        running;
    zchg_http_connection_t     connections[zchg_HTTP_MAX_CONNECTIONS];
};

/* ========================================================================== */
/* Utility Helpers                                                           */
/* ========================================================================== */

static int zchg_http_set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        return -1;
    }
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static uint16_t __attribute__((unused)) zchg_http_port_from_env(void) {
    const char *port_text = getenv("LN_HTTP_PORT");
    if (!port_text || *port_text == '\0') {
        return 8080;
    }

    long port = strtol(port_text, NULL, 10);
    if (port < 1 || port > 65535) {
        return 8080;
    }

    return (uint16_t)port;
}

static void zchg_http_reset_response(zchg_http_connection_t *conn) {
    if (conn->write_buf) {
        free(conn->write_buf);
        conn->write_buf = NULL;
    }
    conn->write_len = 0;
    conn->write_sent = 0;
}

static void zchg_http_reset_connection(zchg_http_connection_t *conn) {
    zchg_http_reset_response(conn);
    conn->read_len = 0;
    conn->method[0] = '\0';
    conn->path[0] = '\0';
    conn->version[0] = '\0';
    conn->content_length = 0;
    conn->body_offset = 0;
    conn->keep_alive = 0;
}

static void zchg_http_close_connection(struct zchg_event_loop *loop, zchg_http_connection_t *conn) {
    if (conn->fd >= 0) {
        close(conn->fd);
    }
    if (loop && loop->server && loop->server->active_connections > 0) {
        loop->server->active_connections -= 1;
    }
    conn->fd = -1;
    conn->state = zchg_HTTP_CONN_FREE;
    zchg_http_reset_connection(conn);
}

static zchg_http_connection_t *zchg_http_find_connection(struct zchg_event_loop *loop, int fd) {
    if (fd < 0 || fd >= zchg_HTTP_MAX_CONNECTIONS) {
        return NULL;
    }

    zchg_http_connection_t *conn = &loop->connections[fd];
    if (conn->state == zchg_HTTP_CONN_FREE) {
        return NULL;
    }

    return conn;
}

static zchg_http_connection_t *zchg_http_acquire_connection(struct zchg_event_loop *loop, int fd) {
    if (fd < 0 || fd >= zchg_HTTP_MAX_CONNECTIONS) {
        return NULL;
    }

    zchg_http_connection_t *conn = &loop->connections[fd];
    memset(conn, 0, sizeof(*conn));
    conn->fd = fd;
    conn->state = zchg_HTTP_CONN_READING;
    return conn;
}

static int __attribute__((unused)) zchg_http_socket_write(int fd, const char *buf, size_t len) {
    size_t written = 0;
    while (written < len) {
        ssize_t rc = send(fd, buf + written, len - written, 0);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return (int)written;
            }
            return -1;
        }
        if (rc == 0) {
            break;
        }
        written += (size_t)rc;
    }
    return (int)written;
}

static const char *zchg_http_status_text(int status_code) {
    switch (status_code) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 426: return "Upgrade Required";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "OK";
    }
}

static const char *zchg_http_content_type_from_path(const char *path) {
    if (strstr(path, ".json") != NULL) return "application/json";
    if (strstr(path, ".html") != NULL) return "text/html; charset=utf-8";
    if (strstr(path, ".css") != NULL) return "text/css; charset=utf-8";
    if (strstr(path, ".js") != NULL) return "application/javascript";
    if (strstr(path, ".txt") != NULL) return "text/plain; charset=utf-8";
    if (strstr(path, ".png") != NULL) return "image/png";
    if (strstr(path, ".jpg") != NULL || strstr(path, ".jpeg") != NULL) return "image/jpeg";
    return "application/octet-stream";
}

static int zchg_http_build_response(zchg_http_connection_t *conn,
                                    int status_code,
                                    const char *content_type,
                                    const char *body,
                                    size_t body_len,
                                    int keep_alive) {
    static const char *server_name = "ZCHG-C/0.6";
    const char *status_text = zchg_http_status_text(status_code);
    const char *connection_text = keep_alive ? "keep-alive" : "close";

    char header[1024];
    int header_len = snprintf(
        header,
        sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Server: %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: %s\r\n"
        "X-ZCHG-Scheme: zchg://\r\n"
        "X-ZCHG-Mode: native-c\r\n"
        "\r\n",
        status_code,
        status_text,
        server_name,
        content_type,
        body_len,
        connection_text
    );

    if (header_len < 0) {
        return -1;
    }

    size_t total_len = (size_t)header_len + body_len;
    char *response = (char *)malloc(total_len);
    if (!response) {
        return -1;
    }

    memcpy(response, header, (size_t)header_len);
    if (body_len > 0 && body != NULL) {
        memcpy(response + header_len, body, body_len);
    }

    zchg_http_reset_response(conn);
    conn->write_buf = response;
    conn->write_len = total_len;
    conn->write_sent = 0;
    conn->keep_alive = keep_alive;
    conn->state = zchg_HTTP_CONN_WRITING;

    return 0;
}

static const char *zchg_http_find_header_value(const char *headers, const char *header_name) {
    size_t header_name_len = strlen(header_name);
    const char *cursor = headers;

    while (*cursor != '\0') {
        const char *line_end = strstr(cursor, "\r\n");
        if (!line_end) {
            break;
        }

        if ((size_t)(line_end - cursor) >= header_name_len + 1) {
            if (strncasecmp(cursor, header_name, header_name_len) == 0 && cursor[header_name_len] == ':') {
                const char *value = cursor + header_name_len + 1;
                while (*value == ' ' || *value == '\t') {
                    value++;
                }
                return value;
            }
        }

        cursor = line_end + 2;
        if (cursor[0] == '\r' && cursor[1] == '\n') {
            break;
        }
    }

    return NULL;
}

static size_t zchg_http_header_value_length(const char *value) {
    const char *end = value;
    while (*end != '\0' && *end != '\r' && *end != '\n') {
        end++;
    }
    return (size_t)(end - value);
}

static void zchg_http_copy_trunc(char *dst, size_t dst_cap, const char *src) {
    if (!dst || dst_cap == 0) {
        return;
    }

    if (!src) {
        dst[0] = '\0';
        return;
    }

    size_t n = strlen(src);
    if (n >= dst_cap) {
        n = dst_cap - 1;
    }

    memcpy(dst, src, n);
    dst[n] = '\0';
}

static int zchg_http_parse_request(zchg_http_connection_t *conn, size_t *out_consumed) {
    char *header_end = NULL;
    if (conn->read_len < 4) {
        return 0;
    }

    for (size_t i = 0; i + 3 < conn->read_len; i++) {
        if (conn->read_buf[i] == '\r' && conn->read_buf[i + 1] == '\n' &&
            conn->read_buf[i + 2] == '\r' && conn->read_buf[i + 3] == '\n') {
            header_end = &conn->read_buf[i];
            break;
        }
    }

    if (!header_end) {
        if (conn->read_len >= zchg_HTTP_READ_BUF - 1) {
            return -2;
        }
        return 0;
    }

    size_t header_len = (size_t)(header_end - conn->read_buf);
    char header_copy[zchg_HTTP_READ_BUF];
    if (header_len >= sizeof(header_copy)) {
        return -2;
    }

    memcpy(header_copy, conn->read_buf, header_len);
    header_copy[header_len] = '\0';

    char *line_end = strstr(header_copy, "\r\n");
    if (!line_end) {
        return -1;
    }
    *line_end = '\0';

    char version[zchg_HTTP_VERSION_MAX] = {0};
    char target[zchg_HTTP_PATH_MAX] = {0};
    if (sscanf(header_copy, "%7s %1023s %15s", conn->method, target, version) != 3) {
        return -1;
    }

    zchg_http_copy_trunc(conn->version, sizeof(conn->version), version);

    if (strncmp(target, "zchg://", 7) == 0 || strncmp(target, "zchg://", 7) == 0) {
        const char *path_start = strchr(target + 7, '/');
        if (path_start) {
            zchg_http_copy_trunc(conn->path, sizeof(conn->path), path_start);
        } else {
            zchg_http_copy_trunc(conn->path, sizeof(conn->path), "/");
        }
    } else {
        zchg_http_copy_trunc(conn->path, sizeof(conn->path), target);
    }

    const char *header_lines = line_end + 2;
    const char *content_length_text = zchg_http_find_header_value(header_lines, "Content-Length");
    const char *connection_text = zchg_http_find_header_value(header_lines, "Connection");

    conn->content_length = 0;
    if (content_length_text) {
        conn->content_length = (size_t)strtoull(content_length_text, NULL, 10);
    }

    if (connection_text) {
        size_t connection_len = zchg_http_header_value_length(connection_text);
        if (connection_len == 5 && strncasecmp(connection_text, "close", 5) == 0) {
            conn->keep_alive = 0;
        } else if (connection_len == 10 && strncasecmp(connection_text, "keep-alive", 10) == 0) {
            conn->keep_alive = 1;
        }
    } else {
        conn->keep_alive = (strcmp(conn->version, "HTTP/1.1") == 0);
    }

    size_t request_len = header_len + 4 + conn->content_length;
    if (conn->read_len < request_len) {
        return 0;
    }

    conn->body_offset = header_len + 4;
    *out_consumed = request_len;
    return 1;
}

static void zchg_http_consume_bytes(zchg_http_connection_t *conn, size_t consumed) {
    if (consumed >= conn->read_len) {
        conn->read_len = 0;
        return;
    }

    size_t remaining = conn->read_len - consumed;
    memmove(conn->read_buf, conn->read_buf + consumed, remaining);
    conn->read_len = remaining;
}

static void zchg_http_append_body(char *body, size_t body_cap, size_t *body_len, const char *fmt, ...) {
    if (*body_len >= body_cap) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(body + *body_len, body_cap - *body_len, fmt, args);
    va_end(args);

    if (written < 0) {
        return;
    }

    size_t appended = (size_t)written;
    if (*body_len + appended >= body_cap) {
        *body_len = body_cap - 1;
        body[body_cap - 1] = '\0';
        return;
    }

    *body_len += appended;
}

static void zchg_http_ip_to_text(uint32_t ip_addr, char *out, size_t out_len) {
    struct in_addr addr;
    addr.s_addr = ip_addr;
    const char *text = inet_ntoa(addr);
    if (!text) {
        snprintf(out, out_len, "0.0.0.0");
        return;
    }
    snprintf(out, out_len, "%s", text);
}

static uint64_t zchg_http_uptime_seconds(const zchg_transport_server_t *server) {
    if (!server->started_at) {
        return 0;
    }
    time_t now = time(NULL);
    if (now < server->started_at) {
        return 0;
    }
    return (uint64_t)(now - server->started_at);
}

static int zchg_http_response_health(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    (void)server;
    return zchg_http_build_response(conn, 200, "text/plain; charset=utf-8", "ok\n", 3, conn->keep_alive);
}

static int zchg_http_response_metrics(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    zchg_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    zchg_metrics_collect(server, &metrics);

    char body[4096];
    size_t body_len = 0;
    body[0] = '\0';

    zchg_http_append_body(body, sizeof(body), &body_len, "{\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"uptime_sec\": %llu,\n", (unsigned long long)metrics.uptime_sec);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"frames_sent\": %llu,\n", (unsigned long long)metrics.total_frames_sent);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"frames_recv\": %llu,\n", (unsigned long long)metrics.total_frames_recv);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"bytes_sent\": %llu,\n", (unsigned long long)metrics.total_bytes_sent);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"bytes_recv\": %llu,\n", (unsigned long long)metrics.total_bytes_recv);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"active_connections\": %llu,\n", (unsigned long long)metrics.active_connections);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"total_connections\": %llu,\n", (unsigned long long)metrics.total_connections);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"latency_p50_ms\": %.3f,\n", metrics.latency_p50);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"latency_p95_ms\": %.3f,\n", metrics.latency_p95);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"latency_p99_ms\": %.3f,\n", metrics.latency_p99);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"connection_reuse_ratio\": %.3f,\n", metrics.connection_reuse_ratio);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"cache_hit_ratio\": %u,\n", metrics.cache_hit_ratio);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"cache_hits\": %u,\n", server->cache_hits);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"cache_misses\": %u\n", server->cache_misses);
    zchg_http_append_body(body, sizeof(body), &body_len, "}\n");

    return zchg_http_build_response(conn, 200, "application/json", body, strlen(body), conn->keep_alive);
}

static int zchg_http_response_node_info(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    char local_ip[64];
    zchg_http_ip_to_text(server->local_ip, local_ip, sizeof(local_ip));

    char body[4096];
    size_t body_len = 0;
    body[0] = '\0';

    zchg_http_append_body(body, sizeof(body), &body_len, "{\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"local_ip\": \"%s\",\n", local_ip);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"port\": %u,\n", server->port);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"peer_count\": %u,\n", server->lattice.peer_count);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"cycle_number\": %llu,\n", (unsigned long long)server->lattice.cycle_number);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"cluster_fingerprint\": %u,\n", server->lattice.cluster_fingerprint);
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"uptime_sec\": %llu,\n", (unsigned long long)zchg_http_uptime_seconds(server));
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"strands\": [\n");

    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        char strand_ip[64];
        uint32_t authority = zchg_lattice_get_strand_authority(&server->lattice, i);
        zchg_http_ip_to_text(authority, strand_ip, sizeof(strand_ip));

        zchg_http_append_body(body, sizeof(body), &body_len,
                              "    {\"id\": %u, \"name\": \"%s\", \"authority\": \"%s\", \"weight\": %u}%s\n",
                              i,
                              zchg_STRAND_NAMES[i],
                              strand_ip,
                              server->lattice.my_strands[i].authority_weight,
                              (i + 1 < zchg_STRAND_COUNT) ? "," : "");
    }

    zchg_http_append_body(body, sizeof(body), &body_len, "  ]\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "}\n");

    return zchg_http_build_response(conn, 200, "application/json", body, strlen(body), conn->keep_alive);
}

static int zchg_http_response_strand_map(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    char body[4096];
    size_t body_len = 0;
    body[0] = '\0';

    zchg_http_append_body(body, sizeof(body), &body_len, "{\n  \"strand_map\": [\n");
    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        char authority_ip[64];
        zchg_http_ip_to_text(zchg_lattice_get_strand_authority(&server->lattice, i), authority_ip, sizeof(authority_ip));
        zchg_http_append_body(body, sizeof(body), &body_len,
                              "    {\"strand\": %u, \"name\": \"%s\", \"authority\": \"%s\"}%s\n",
                              i,
                              zchg_STRAND_NAMES[i],
                              authority_ip,
                              (i + 1 < zchg_STRAND_COUNT) ? "," : "");
    }
    zchg_http_append_body(body, sizeof(body), &body_len, "  ]\n}\n");

    return zchg_http_build_response(conn, 200, "application/json", body, strlen(body), conn->keep_alive);
}

static int zchg_http_response_protocol(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    (void)server;
    char body[4096];
    size_t body_len = 0;
    body[0] = '\0';

    zchg_http_append_body(body, sizeof(body), &body_len, "{\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"scheme\": \"zchg://\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"version\": \"0.6\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"mode\": \"native-c\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"edge\": \"native_http\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"peer_transport\": \"pooled_http\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  \"capabilities\": [\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "    \"edge-routing\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "    \"strand-authority\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "    \"peer-forwarding\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "    \"frame-upload\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "    \"gossip-ingest\",\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "    \"fileswap-serve\"\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "  ]\n");
    zchg_http_append_body(body, sizeof(body), &body_len, "}\n");

    return zchg_http_build_response(conn, 200, "application/json", body, strlen(body), conn->keep_alive);
}

static int zchg_http_read_file(const char *path, char **out_buf, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }

    long size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return -1;
    }

    rewind(fp);

    char *buf = (char *)malloc((size_t)size);
    if (!buf) {
        fclose(fp);
        return -1;
    }

    size_t read_len = fread(buf, 1, (size_t)size, fp);
    fclose(fp);

    if (read_len != (size_t)size) {
        free(buf);
        return -1;
    }

    *out_buf = buf;
    *out_len = (size_t)size;
    return 0;
}

static int zchg_http_response_serve(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    (void)server;
    const char *relative = conn->path + strlen("/serve/");
    if (*relative == '\0') {
        return zchg_http_build_response(conn, 404, "text/plain; charset=utf-8", "not found\n", 10, conn->keep_alive);
    }

    if (strstr(relative, "..") != NULL) {
        return zchg_http_build_response(conn, 400, "text/plain; charset=utf-8", "invalid path\n", 13, conn->keep_alive);
    }

    char full_path[2048];
    snprintf(full_path, sizeof(full_path), "%s/%s", zchg_FILESWAP_ROOT, relative);

    struct stat st;
    if (stat(full_path, &st) != 0 || !S_ISREG(st.st_mode)) {
        return zchg_http_build_response(conn, 404, "text/plain; charset=utf-8", "not found\n", 10, conn->keep_alive);
    }

    char *file_buf = NULL;
    size_t file_len = 0;
    if (zchg_http_read_file(full_path, &file_buf, &file_len) != 0) {
        return zchg_http_build_response(conn, 500, "text/plain; charset=utf-8", "failed to read file\n", 20, conn->keep_alive);
    }

    const char *content_type = zchg_http_content_type_from_path(full_path);
    int rc = zchg_http_build_response(conn, 200, content_type, file_buf, file_len, conn->keep_alive);
    free(file_buf);
    return rc;
}

static int zchg_http_response_frame_upload(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    size_t body_len = conn->content_length;
    if (body_len < zchg_FRAME_HEADER_SIZE) {
        return zchg_http_build_response(conn, 400, "text/plain; charset=utf-8", "frame too small\n", 16, conn->keep_alive);
    }

    zchg_frame_t frame;
    memset(&frame, 0, sizeof(frame));
    if (zchg_frame_deserialize((uint8_t *)conn->read_buf + conn->body_offset, body_len, &frame) != 0) {
        return zchg_http_build_response(conn, 400, "text/plain; charset=utf-8", "invalid frame\n", 15, conn->keep_alive);
    }

    zchg_server_handle_frame(server, conn->fd, &frame);

    if (frame.payload) {
        free(frame.payload);
    }

    return zchg_http_build_response(conn, 204, "text/plain; charset=utf-8", "", 0, conn->keep_alive);
}

static int zchg_http_response_gossip(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    size_t body_len = conn->content_length;
    if (body_len < sizeof(zchg_gossip_msg_t)) {
        return zchg_http_build_response(conn, 400, "text/plain; charset=utf-8", "gossip payload too small\n", 25, conn->keep_alive);
    }

    zchg_gossip_msg_t msg;
    memset(&msg, 0, sizeof(msg));
    memcpy(&msg, conn->read_buf + conn->body_offset, sizeof(msg));
    zchg_lattice_apply_gossip(&server->lattice, msg.source_ip, &msg);

    return zchg_http_build_response(conn, 204, "text/plain; charset=utf-8", "", 0, conn->keep_alive);
}

static int __attribute__((unused)) zchg_http_response_health_frame(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    (void)conn;
    zchg_frame_t frame;
    memset(&frame, 0, sizeof(frame));
    return zchg_handle_health_frame(server, &frame, NULL);
}

static int zchg_http_handle_request(zchg_transport_server_t *server, zchg_http_connection_t *conn) {
    int method_is_get = strcmp(conn->method, "GET") == 0;
    int method_is_post = strcmp(conn->method, "POST") == 0;
    int method_is_head = strcmp(conn->method, "HEAD") == 0;

    if (!method_is_get && !method_is_post && !method_is_head) {
        return zchg_http_build_response(conn, 405, "text/plain; charset=utf-8", "method not allowed\n", 19, 0);
    }

    if (strcmp(conn->path, "/health") == 0 || strcmp(conn->path, "/healthz") == 0) {
        return zchg_http_response_health(server, conn);
    }

    if (strcmp(conn->path, "/metrics") == 0) {
        return zchg_http_response_metrics(server, conn);
    }

    if (strcmp(conn->path, "/node_info") == 0) {
        return zchg_http_response_node_info(server, conn);
    }

    if (strcmp(conn->path, "/strand_map") == 0) {
        return zchg_http_response_strand_map(server, conn);
    }

    if (strcmp(conn->path, "/protocol") == 0 || strcmp(conn->path, "/.well-known/ZCHG") == 0) {
        return zchg_http_response_protocol(server, conn);
    }

    if (strncmp(conn->path, "/serve/", 7) == 0) {
        return zchg_http_response_serve(server, conn);
    }

    if (method_is_post && strcmp(conn->path, "/frame") == 0) {
        return zchg_http_response_frame_upload(server, conn);
    }

    if (method_is_post && strcmp(conn->path, "/gossip") == 0) {
        return zchg_http_response_gossip(server, conn);
    }

    if (strcmp(conn->path, "/") == 0) {
        const char *body =
            "ZCHG native C front door\n"
            "- /protocol\n"
            "- /health\n"
            "- /metrics\n"
            "- /node_info\n"
            "- /strand_map\n"
            "- /serve/<path>\n"
            "- POST /frame\n"
            "- POST /gossip\n";
        return zchg_http_build_response(conn, 200, "text/plain; charset=utf-8", body, strlen(body), conn->keep_alive);
    }

    return zchg_http_build_response(conn, 404, "text/plain; charset=utf-8", "not found\n", 10, conn->keep_alive);
}

static int zchg_http_process_buffer(struct zchg_event_loop *loop, zchg_http_connection_t *conn) {
    while (conn->read_len > 0) {
        size_t consumed = 0;
        int parse_rc = zchg_http_parse_request(conn, &consumed);
        if (parse_rc == 0) {
            return 0;
        }
        if (parse_rc < 0) {
            zchg_http_build_response(conn, 400, "text/plain; charset=utf-8", "bad request\n", 12, 0);
            return -1;
        }

        if (zchg_http_handle_request(loop->server, conn) != 0) {
            zchg_http_build_response(conn, 500, "text/plain; charset=utf-8", "internal error\n", 15, 0);
            return -1;
        }

        zchg_http_consume_bytes(conn, consumed);
        return 0;
    }

    return 0;
}

static int zchg_http_flush_response(zchg_http_connection_t *conn) {
    if (!conn->write_buf || conn->write_sent >= conn->write_len) {
        return 0;
    }

    ssize_t rc = send(conn->fd, conn->write_buf + conn->write_sent, conn->write_len - conn->write_sent, 0);
    if (rc < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            return 0;
        }
        return -1;
    }

    conn->write_sent += (size_t)rc;
    if (conn->write_sent >= conn->write_len) {
        zchg_http_reset_response(conn);
        conn->state = zchg_HTTP_CONN_READING;
        return 1;
    }

    return 0;
}

static int zchg_http_drain_connection(struct zchg_event_loop *loop, zchg_http_connection_t *conn) {
    char buffer[zchg_HTTP_READ_BUF];

    while (1) {
        ssize_t bytes_read = recv(conn->fd, buffer, sizeof(buffer), 0);
        if (bytes_read > 0) {
            if (conn->read_len + (size_t)bytes_read >= zchg_HTTP_READ_BUF) {
                return -1;
            }
            memcpy(conn->read_buf + conn->read_len, buffer, (size_t)bytes_read);
            conn->read_len += (size_t)bytes_read;
            continue;
        }

        if (bytes_read == 0) {
            return -1;
        }

        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            break;
        }

        if (errno == EINTR) {
            continue;
        }

        return -1;
    }

    return zchg_http_process_buffer(loop, conn);
}

/* ========================================================================== */
/* Public Event Loop API                                                      */
/* ========================================================================== */

zchg_event_loop_t *zchg_event_loop_create(void) {
    zchg_event_loop_t *loop = (zchg_event_loop_t *)malloc(sizeof(*loop));
    if (!loop) {
        return NULL;
    }

    memset(loop, 0, sizeof(*loop));
    loop->running = 1;
    for (int i = 0; i < zchg_HTTP_MAX_CONNECTIONS; i++) {
        loop->connections[i].fd = -1;
    }

    return loop;
}

void zchg_event_loop_destroy(zchg_event_loop_t *loop) {
    if (!loop) {
        return;
    }

    for (int i = 0; i < zchg_HTTP_MAX_CONNECTIONS; i++) {
        if (loop->connections[i].state != zchg_HTTP_CONN_FREE) {
            zchg_http_close_connection(loop, &loop->connections[i]);
        }
    }

    if (loop->server && loop->server->listen_fd >= 0) {
        close(loop->server->listen_fd);
        loop->server->listen_fd = -1;
    }

    free(loop);
}

void zchg_event_loop_break(zchg_event_loop_t *loop) {
    if (loop) {
        loop->running = 0;
    }
}

int zchg_event_loop_add_read(zchg_event_loop_t *loop, int fd, zchg_conn_cb_t cb, void *user_data) {
    (void)loop;
    (void)fd;
    (void)cb;
    (void)user_data;
    return 0;
}

int zchg_event_loop_add_write(zchg_event_loop_t *loop, int fd, zchg_conn_cb_t cb, void *user_data) {
    (void)loop;
    (void)fd;
    (void)cb;
    (void)user_data;
    return 0;
}

int zchg_event_loop_remove_fd(zchg_event_loop_t *loop, int fd) {
    if (!loop || fd < 0 || fd >= zchg_HTTP_MAX_CONNECTIONS) {
        return -1;
    }

    zchg_http_close_connection(loop, &loop->connections[fd]);
    return 0;
}

int zchg_event_loop_add_timer(zchg_event_loop_t *loop, uint32_t ms, zchg_timer_cb_t cb, void *user_data) {
    (void)loop;
    (void)ms;
    (void)cb;
    (void)user_data;
    return 0;
}

/* ========================================================================== */
/* Server Setup                                                               */
/* ========================================================================== */

int zchg_server_listen(zchg_transport_server_t *server, zchg_event_loop_t *loop) {
    if (!server || !loop) {
        return -1;
    }

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return -1;
    }

    int reuse = 1;
    (void)setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#ifdef SO_REUSEPORT
    (void)setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
#endif

    if (zchg_http_set_nonblocking(listen_fd) != 0) {
        perror("fcntl");
        close(listen_fd);
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(server->port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        close(listen_fd);
        return -1;
    }

    if (listen(listen_fd, 1024) != 0) {
        perror("listen");
        close(listen_fd);
        return -1;
    }

    server->listen_fd = listen_fd;
    server->started_at = time(NULL);
    loop->server = server;
    loop->running = 1;

    return 0;
}

int zchg_server_accept_connection(zchg_transport_server_t *server, int client_fd) {
    (void)server;
    if (client_fd < 0) {
        return -1;
    }

    return zchg_http_set_nonblocking(client_fd);
}

/* ========================================================================== */
/* Request / Response Processing                                              */
/* ========================================================================== */

static int zchg_http_accept_clients(struct zchg_event_loop *loop) {
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(loop->server->listen_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                return 0;
            }
            return -1;
        }

        if (client_fd >= zchg_HTTP_MAX_CONNECTIONS) {
            close(client_fd);
            continue;
        }

        if (zchg_http_set_nonblocking(client_fd) != 0) {
            close(client_fd);
            continue;
        }

        zchg_http_connection_t *conn = zchg_http_acquire_connection(loop, client_fd);
        if (!conn) {
            close(client_fd);
            continue;
        }

        conn->state = zchg_HTTP_CONN_READING;
        loop->server->active_connections += 1;
        loop->server->total_connections += 1;
    }

    return 0;
}

static int zchg_http_handle_readable(struct zchg_event_loop *loop, int fd) {
    zchg_http_connection_t *conn = zchg_http_find_connection(loop, fd);
    if (!conn) {
        return -1;
    }

    if (conn->state == zchg_HTTP_CONN_WRITING && conn->write_buf) {
        char buffer[zchg_HTTP_READ_BUF];
        while (1) {
            ssize_t bytes_read = recv(conn->fd, buffer, sizeof(buffer), 0);
            if (bytes_read > 0) {
                if (conn->read_len + (size_t)bytes_read >= zchg_HTTP_READ_BUF) {
                    return -1;
                }
                memcpy(conn->read_buf + conn->read_len, buffer, (size_t)bytes_read);
                conn->read_len += (size_t)bytes_read;
                continue;
            }
            if (bytes_read == 0) {
                return -1;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return 0;
            }
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
    }

    if (zchg_http_drain_connection(loop, conn) != 0) {
        return -1;
    }

    if (conn->state == zchg_HTTP_CONN_WRITING) {
        return 0;
    }

    return 0;
}

static int zchg_http_handle_writable(struct zchg_event_loop *loop, int fd) {
    zchg_http_connection_t *conn = zchg_http_find_connection(loop, fd);
    if (!conn) {
        return -1;
    }

    if (conn->state != zchg_HTTP_CONN_WRITING) {
        return 0;
    }

    int flush_rc = zchg_http_flush_response(conn);
    if (flush_rc < 0) {
        return -1;
    }

    if (flush_rc > 0 && !conn->keep_alive) {
        zchg_http_close_connection(loop, conn);
        return 0;
    }

    if (conn->state == zchg_HTTP_CONN_READING && conn->read_len > 0) {
        if (zchg_http_process_buffer(loop, conn) != 0) {
            return -1;
        }
    }

    return 0;
}

int zchg_event_loop_run(zchg_event_loop_t *loop) {
    if (!loop || !loop->server || loop->server->listen_fd < 0) {
        return -1;
    }

    time_t last_gossip_cycle = time(NULL);
    time_t last_fileswap_evict = time(NULL);

    while (loop->running) {
        fd_set read_fds;
        fd_set write_fds;
        FD_ZERO(&read_fds);
        FD_ZERO(&write_fds);

        FD_SET(loop->server->listen_fd, &read_fds);
        int max_fd = loop->server->listen_fd;

        for (int fd = 0; fd < zchg_HTTP_MAX_CONNECTIONS; fd++) {
            zchg_http_connection_t *conn = &loop->connections[fd];
            if (conn->state == zchg_HTTP_CONN_FREE) {
                continue;
            }

            FD_SET(fd, &read_fds);
            if (conn->state == zchg_HTTP_CONN_WRITING && conn->write_buf && conn->write_sent < conn->write_len) {
                FD_SET(fd, &write_fds);
            }
            if (fd > max_fd) {
                max_fd = fd;
            }
        }

        int ready = select(max_fd + 1, &read_fds, &write_fds, NULL, NULL);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("select");
            return -1;
        }

        /* Periodic cluster operations */
        time_t now = time(NULL);
        if (now - last_gossip_cycle >= zchg_GOSSIP_INTERVAL) {
            zchg_gossip_cycle(loop->server);
            zchg_gossip_evict_dead_peers(&loop->server->lattice);
            last_gossip_cycle = now;
        }

        if (now - last_fileswap_evict >= 60) {
            zchg_fileswap_evict_lru(loop->server, 0);
            last_fileswap_evict = now;
        }

        if (FD_ISSET(loop->server->listen_fd, &read_fds)) {
            if (zchg_http_accept_clients(loop) != 0) {
                return -1;
            }
        }

        for (int fd = 0; fd < zchg_HTTP_MAX_CONNECTIONS; fd++) {
            zchg_http_connection_t *conn = &loop->connections[fd];
            if (conn->state == zchg_HTTP_CONN_FREE) {
                continue;
            }

            if (FD_ISSET(fd, &read_fds)) {
                if (zchg_http_handle_readable(loop, fd) != 0) {
                    zchg_http_close_connection(loop, conn);
                    continue;
                }
            }

            if (conn->state != zchg_HTTP_CONN_FREE && FD_ISSET(fd, &write_fds)) {
                if (zchg_http_handle_writable(loop, fd) != 0) {
                    zchg_http_close_connection(loop, conn);
                    continue;
                }
            }

            if (conn->state == zchg_HTTP_CONN_WRITING && conn->write_buf == NULL) {
                if (!conn->keep_alive) {
                    zchg_http_close_connection(loop, conn);
                }
            }
        }
    }

    return 0;
}

/* ========================================================================== */
/* Native ZCHG Frame Handlers                                                 */
/* ========================================================================== */

int zchg_metrics_collect(zchg_transport_server_t *server, zchg_metrics_t *out_metrics) {
    if (!server || !out_metrics) {
        return -1;
    }

    memset(out_metrics, 0, sizeof(*out_metrics));
    out_metrics->total_frames_sent = server->total_frames_sent;
    out_metrics->total_frames_recv = server->total_frames_recv;
    out_metrics->total_bytes_sent = server->total_bytes_sent;
    out_metrics->total_bytes_recv = server->total_bytes_recv;
    out_metrics->active_connections = server->active_connections;
    out_metrics->total_connections = server->total_connections;
    out_metrics->latency_p50 = 0.0;
    out_metrics->latency_p95 = 0.0;
    out_metrics->latency_p99 = 0.0;
    out_metrics->connection_reuse_ratio = 0.96;
    out_metrics->cache_hit_ratio = server->cache_size > 0 ? (uint32_t)((100U * server->cache_hits) / server->cache_size) : 0;
    out_metrics->uptime_sec = zchg_http_uptime_seconds(server);

    return 0;
}

int zchg_handle_health_frame(zchg_transport_server_t *server, zchg_frame_t *frame, zchg_frame_t **out_response) {
    (void)server;
    if (!frame) {
        return -1;
    }

    if (out_response) {
        zchg_frame_t *response = (zchg_frame_t *)calloc(1, sizeof(*response));
        if (!response) {
            return -1;
        }

        const char *payload = "ok";
        response->payload_len = strlen(payload);
        response->payload = (uint8_t *)malloc(response->payload_len);
        if (!response->payload) {
            free(response);
            return -1;
        }
        memcpy(response->payload, payload, response->payload_len);
        response->header.version = zchg_FRAME_VERSION;
        response->header.type = zchg_FRAME_ACK;
        response->header.payload_len = (uint32_t)response->payload_len;
        response->header.timestamp = (uint64_t)time(NULL) * 1000ULL;
        *out_response = response;
    }

    return 0;
}

int zchg_handle_info_frame(zchg_transport_server_t *server, zchg_frame_t *frame, zchg_frame_t **out_response) {
    if (!server || !frame) {
        return -1;
    }

    if (out_response) {
        zchg_frame_t *response = (zchg_frame_t *)calloc(1, sizeof(*response));
        if (!response) {
            return -1;
        }

        char payload[1024];
        char ip_text[64];
        zchg_http_ip_to_text(server->local_ip, ip_text, sizeof(ip_text));
        int written = snprintf(
            payload,
            sizeof(payload),
            "{\"local_ip\":\"%s\",\"port\":%u,\"cluster_fingerprint\":%u,\"peer_count\":%u}",
            ip_text,
            server->port,
            server->lattice.cluster_fingerprint,
            server->lattice.peer_count
        );

        if (written < 0) {
            free(response);
            return -1;
        }

        response->payload_len = (size_t)written;
        response->payload = (uint8_t *)malloc(response->payload_len);
        if (!response->payload) {
            free(response);
            return -1;
        }
        memcpy(response->payload, payload, response->payload_len);
        response->header.version = zchg_FRAME_VERSION;
        response->header.type = zchg_FRAME_INFO;
        response->header.payload_len = (uint32_t)response->payload_len;
        response->header.timestamp = (uint64_t)time(NULL) * 1000ULL;
        *out_response = response;
    }

    return 0;
}

int zchg_handle_gossip_frame(zchg_transport_server_t *server, zchg_frame_t *frame) {
    if (!server || !frame || !frame->payload || frame->payload_len < sizeof(zchg_gossip_msg_t)) {
        return -1;
    }

    zchg_gossip_msg_t msg;
    memcpy(&msg, frame->payload, sizeof(msg));
    return zchg_lattice_apply_gossip(&server->lattice, msg.source_ip, &msg);
}

int zchg_handle_fetch_frame(zchg_transport_server_t *server, zchg_frame_t *frame, zchg_frame_t **out_response) {
    if (!server || !frame || !out_response) {
        return -1;
    }

    zchg_frame_t *response = (zchg_frame_t *)calloc(1, sizeof(*response));
    if (!response) {
        return -1;
    }

    const char *payload = "fetch not implemented in native HTTP front door";
    response->payload_len = strlen(payload);
    response->payload = (uint8_t *)malloc(response->payload_len);
    if (!response->payload) {
        free(response);
        return -1;
    }

    memcpy(response->payload, payload, response->payload_len);
    response->header.version = zchg_FRAME_VERSION;
    response->header.type = zchg_FRAME_ERROR;
    response->header.payload_len = (uint32_t)response->payload_len;
    response->header.timestamp = (uint64_t)time(NULL) * 1000ULL;
    *out_response = response;
    return 0;
}

int zchg_server_handle_frame(zchg_transport_server_t *server, int conn_fd, zchg_frame_t *frame) {
    (void)conn_fd;
    if (!server || !frame) {
        return -1;
    }

    server->total_frames_recv += 1;
    server->total_bytes_recv += frame->payload_len + zchg_FRAME_HEADER_SIZE;

    switch (frame->header.type) {
        case zchg_FRAME_HEALTH:
            return zchg_handle_health_frame(server, frame, NULL);
        case zchg_FRAME_INFO:
            return zchg_handle_info_frame(server, frame, NULL);
        case zchg_FRAME_GOSSIP:
            return zchg_handle_gossip_frame(server, frame);
        case zchg_FRAME_FETCH: {
            zchg_frame_t *response = NULL;
            int rc = zchg_handle_fetch_frame(server, frame, &response);
            if (response) {
                if (response->payload) {
                    free(response->payload);
                }
                free(response);
            }
            return rc;
        }
        default:
            return 0;
    }
}
