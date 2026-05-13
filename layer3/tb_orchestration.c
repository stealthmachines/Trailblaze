/*
 * tb_orchestration.c — TRAILBLAZE Layer 3: Agent Orchestration implementation
 *
 * 19 built-in tools, task compiler (keyword → TB_PassType), graph executor,
 * tb_unfold() top-level, agent fabric delegation, and HTTP extension handler
 * for /unfold /tools /state /ledger /sse.
 */

#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#endif
#include "tb_orchestration.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifdef _WIN32
#  include "../src/tb_win32.h"
#else
#  include <unistd.h>
#  include <sys/stat.h>
#  include <dirent.h>
#endif

/* ── Wall clock (ms) ────────────────────────────────────────────────────── */
static double tb_orch_wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Pass name table ────────────────────────────────────────────────────── */
static const char *tb_pass_name(TB_PassType pt) {
    switch (pt) {
        case TB_PASS_FETCH:     return "FETCH";
        case TB_PASS_SHELL:     return "SHELL";
        case TB_PASS_CODE:      return "CODE";
        case TB_PASS_TRANSFORM: return "TRANSFORM";
        case TB_PASS_STORE:     return "STORE";
        case TB_PASS_RECALL:    return "RECALL";
        case TB_PASS_BROWSE:    return "BROWSE";
        case TB_PASS_NOTIFY:    return "NOTIFY";
        case TB_PASS_RESPOND:   return "RESPOND";
        case TB_PASS_EMBED:     return "EMBED";
        case TB_PASS_ATTEND:    return "ATTEND";
        case TB_PASS_FFN:       return "FFN";
        case TB_PASS_SAMPLE:    return "SAMPLE";
        case TB_PASS_ROUTE:     return "ROUTE";
        case TB_PASS_NORM:      return "NORM";
        case TB_PASS_PROJ:      return "PROJ";
        case TB_PASS_BRANCH:    return "BRANCH";
        case TB_PASS_MERGE:     return "MERGE";
        case TB_PASS_SPECULATE: return "SPECULATE";
        default:                return "UNKNOWN";
    }
}

/* ============================================================================
 * SECTION 1: Simple key-value store helpers
 * ============================================================================ */

#define TB_STORE_CAP 64

typedef struct {
    char keys[TB_STORE_CAP][128];
    char vals[TB_STORE_CAP][2048];
    int  n;
} TB_ExecStore;

static void store_set(TB_ExecStore *s, const char *k, const char *v) {
    for (int i = 0; i < s->n; i++) {
        if (strcmp(s->keys[i], k) == 0) {
            snprintf(s->vals[i], 2048, "%s", v ? v : "");
            return;
        }
    }
    if (s->n < TB_STORE_CAP) {
        snprintf(s->keys[s->n], 128,  "%s", k ? k : "");
        snprintf(s->vals[s->n], 2048, "%s", v ? v : "");
        s->n++;
    }
}

static const char* store_get(TB_ExecStore *s, const char *k) {
    for (int i = 0; i < s->n; i++)
        if (strcmp(s->keys[i], k) == 0) return s->vals[i];
    return NULL;
}

/* ============================================================================
 * SECTION 2: Built-in tool implementations
 * ============================================================================ */

static int tool_shell_exec(const char *args, TB_ToolResult *r, void *ud) {
    (void)ud;
    /* Extract "cmd" from JSON args. */
    const char *cmd = args;
    const char *p = strstr(args, "\"cmd\"");
    if (p) {
        p = strchr(p, ':'); if (p) p++;
        while (*p == ' ' || *p == '"') p++;
        /* Copy until closing quote or end. */
        char cmd_buf[512] = {0};
        int n = 0;
        while (*p && *p != '"' && n < 511) cmd_buf[n++] = *p++;
        cmd_buf[n] = '\0';
        cmd = cmd_buf;
        /* Exec */
        FILE *f = popen(cmd_buf, "r");
        if (!f) {
            snprintf(r->error, sizeof(r->error), "popen failed: %s", cmd_buf);
            return -1;
        }
        int nr = (int)fread(r->data_json, 1, sizeof(r->data_json) - 1, f);
        r->data_json[nr] = '\0';
        pclose(f);
        r->success = 1;
        return 0;
    }
    /* Fallback: treat entire args as command. */
    FILE *f = popen(cmd, "r");
    if (!f) { snprintf(r->error, sizeof(r->error), "popen failed"); return -1; }
    int nr = (int)fread(r->data_json, 1, sizeof(r->data_json) - 1, f);
    r->data_json[nr] = '\0';
    pclose(f);
    r->success = 1;
    return 0;
}

static int tool_read_file(const char *args, TB_ToolResult *r, void *ud) {
    (void)ud;
    char path[512] = {0};
    const char *p = strstr(args, "\"path\"");
    if (p) {
        p = strchr(p, ':'); if (p) p++;
        while (*p == ' ' || *p == '"') p++;
        int n = 0;
        while (*p && *p != '"' && n < 511) path[n++] = *p++;
    } else {
        snprintf(r->error, sizeof(r->error), "missing path");
        return -1;
    }
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(r->error, sizeof(r->error), "fopen failed: %s", path); return -1; }
    int nr = (int)fread(r->data_json, 1, sizeof(r->data_json) - 1, f);
    r->data_json[nr] = '\0';
    fclose(f);
    r->success = 1;
    return 0;
}

static int tool_write_file(const char *args, TB_ToolResult *r, void *ud) {
    (void)ud;
    char path[512] = {0};
    char content[1024] = {0};
    const char *p = strstr(args, "\"path\"");
    if (p) {
        p = strchr(p, ':'); if (p) p++;
        while (*p == ' ' || *p == '"') p++;
        int n = 0; while (*p && *p != '"' && n < 511) path[n++] = *p++;
    }
    const char *c = strstr(args, "\"content\"");
    if (c) {
        c = strchr(c, ':'); if (c) c++;
        while (*c == ' ' || *c == '"') c++;
        int n = 0; while (*c && *c != '"' && n < 1023) content[n++] = *c++;
    }
    FILE *f = fopen(path, "w");
    if (!f) { snprintf(r->error, sizeof(r->error), "fopen write failed: %s", path); return -1; }
    fputs(content, f);
    fclose(f);
    snprintf(r->data_json, sizeof(r->data_json), "{\"written\":%zu}", strlen(content));
    r->success = 1;
    return 0;
}

static int tool_list_dir(const char *args, TB_ToolResult *r, void *ud) {
    (void)ud;
    char path[512] = ".";
    const char *p = strstr(args, "\"path\"");
    if (p) {
        p = strchr(p, ':'); if (p) p++;
        while (*p == ' ' || *p == '"') p++;
        int n = 0; while (*p && *p != '"' && n < 511) path[n++] = *p++;
        path[n] = '\0';
    }
    DIR *d = opendir(path);
    if (!d) { snprintf(r->error, sizeof(r->error), "opendir failed: %s", path); return -1; }
    struct dirent *de;
    char *wp = r->data_json;
    char *end = r->data_json + sizeof(r->data_json) - 2;
    *wp++ = '['; *wp++ = '"';
    int first = 1;
    while ((de = readdir(d)) != NULL && wp < end - 4) {
        if (de->d_name[0] == '.') continue;
        if (!first) { memcpy(wp, "\",\"", 3); wp += 3; }
        first = 0;
        int nl = (int)strlen(de->d_name);
        if (wp + nl < end) { memcpy(wp, de->d_name, nl); wp += nl; }
    }
    closedir(d);
    if (!first) { *wp++ = '"'; }
    *wp++ = ']'; *wp = '\0';
    r->success = 1;
    return 0;
}

static int tool_file_exists(const char *args, TB_ToolResult *r, void *ud) {
    (void)ud;
    char path[512] = {0};
    const char *p = strstr(args, "\"path\"");
    if (p) {
        p = strchr(p, ':'); if (p) p++;
        while (*p == ' ' || *p == '"') p++;
        int n = 0; while (*p && *p != '"' && n < 511) path[n++] = *p++;
    }
    struct stat st;
    int exists = (stat(path, &st) == 0);
    snprintf(r->data_json, sizeof(r->data_json), "{\"exists\":%s,\"path\":\"%s\"}",
             exists ? "true" : "false", path);
    r->success = 1;
    return 0;
}

static int tool_make_dir(const char *args, TB_ToolResult *r, void *ud) {
    (void)ud;
    char path[512] = {0};
    const char *p = strstr(args, "\"path\"");
    if (p) {
        p = strchr(p, ':'); if (p) p++;
        while (*p == ' ' || *p == '"') p++;
        int n = 0; while (*p && *p != '"' && n < 511) path[n++] = *p++;
    }
    int rc = mkdir(path, 0755);
    snprintf(r->data_json, sizeof(r->data_json), "{\"created\":%s}", rc == 0 ? "true" : "false");
    r->success = 1;
    return 0;
}

/* Tree-backed tools — userdata is TB_CognitionTree*. */
static int tool_memory_set(const char *args, TB_ToolResult *r, void *ud) {
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    char key[128] = {0}, val[512] = {0};
    const char *p = strstr(args, "\"key\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' '||*p=='"')p++;
             int n=0; while(*p&&*p!='"'&&n<127)key[n++]=*p++; }
    const char *v = strstr(args, "\"value\"");
    if (v) { v = strchr(v,':'); if(v)v++; while(*v==' '||*v=='"')v++;
             int n=0; while(*v&&*v!='"'&&n<511)val[n++]=*v++; }
    if (!key[0]) { snprintf(r->error,sizeof(r->error),"missing key"); return -1; }
    tb_tree_memory_set(tree, key, val);
    snprintf(r->data_json, sizeof(r->data_json), "{\"set\":\"%s\"}", key);
    r->success = 1;
    return 0;
}

static int tool_memory_get(const char *args, TB_ToolResult *r, void *ud) {
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    char key[128] = {0};
    const char *p = strstr(args, "\"key\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' '||*p=='"')p++;
             int n=0; while(*p&&*p!='"'&&n<127)key[n++]=*p++; }
    const char *val = tb_tree_memory_get(tree, key);
    snprintf(r->data_json, sizeof(r->data_json), "{\"key\":\"%s\",\"value\":\"%s\"}",
             key, val ? val : "");
    r->success = 1;
    return 0;
}

static int tool_notes_write(const char *args, TB_ToolResult *r, void *ud) {
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    char fname[128] = {0}, content[1024] = {0};
    const char *p = strstr(args, "\"filename\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' '||*p=='"')p++;
             int n=0; while(*p&&*p!='"'&&n<127)fname[n++]=*p++; }
    const char *c = strstr(args, "\"content\"");
    if (c) { c = strchr(c,':'); if(c)c++; while(*c==' '||*c=='"')c++;
             int n=0; while(*c&&*c!='"'&&n<1023)content[n++]=*c++; }
    tb_tree_notes_write(tree, fname[0]?fname:"default.md", content);
    snprintf(r->data_json, sizeof(r->data_json), "{\"written\":\"%s\"}", fname);
    r->success = 1;
    return 0;
}

static int tool_notes_read(const char *args, TB_ToolResult *r, void *ud) {
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    char fname[128] = {0};
    const char *p = strstr(args, "\"filename\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' '||*p=='"')p++;
             int n=0; while(*p&&*p!='"'&&n<127)fname[n++]=*p++; }
    char *content = tb_tree_notes_read(tree, fname[0]?fname:"default.md");
    if (content) {
        snprintf(r->data_json, sizeof(r->data_json), "%.*s",
                 (int)sizeof(r->data_json)-1, content);
        free(content);
    } else {
        snprintf(r->data_json, sizeof(r->data_json), "{}");
    }
    r->success = 1;
    return 0;
}

static int tool_branch_create(const char *args, TB_ToolResult *r, void *ud) {
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    int from_id = 0;
    const char *p = strstr(args, "\"from\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' ')p++;
             from_id = (int)strtol(p, NULL, 10); }
    int new_id = tb_tree_branch_create(tree, from_id);
    snprintf(r->data_json, sizeof(r->data_json), "{\"branch_id\":%d}", new_id);
    r->success = (new_id >= 0);
    return new_id >= 0 ? 0 : -1;
}

static int tool_branch_merge(const char *args, TB_ToolResult *r, void *ud) {
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    int src = 0, dst = 0;
    const char *p = strstr(args, "\"src\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' ')p++; src=(int)strtol(p,NULL,10); }
    const char *q = strstr(args, "\"dst\"");
    if (q) { q = strchr(q,':'); if(q)q++; while(*q==' ')q++; dst=(int)strtol(q,NULL,10); }
    int rc = tb_tree_branch_merge(tree, src, dst);
    snprintf(r->data_json, sizeof(r->data_json), "{\"merged\":%s,\"src\":%d,\"dst\":%d}",
             rc==0?"true":"false", src, dst);
    r->success = (rc == 0);
    return rc;
}

static int tool_cell_commit(const char *args, TB_ToolResult *r, void *ud) {
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    int branch_id = 0; char value[512] = {0}, domain[64] = "tool";
    const char *p = strstr(args, "\"branch\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' ')p++; branch_id=(int)strtol(p,NULL,10); }
    const char *v = strstr(args, "\"value\"");
    if (v) { v = strchr(v,':'); if(v)v++; while(*v==' '||*v=='"')v++;
             int n=0; while(*v&&*v!='"'&&n<511)value[n++]=*v++; }
    const char *d = strstr(args, "\"domain\"");
    if (d) { d = strchr(d,':'); if(d)d++; while(*d==' '||*d=='"')d++;
             int n=0; while(*d&&*d!='"'&&n<63)domain[n++]=*d++; }
    TB_CognitionCell *cell = tb_tree_cell_commit(tree, branch_id,
                                                   value[0]?value:"{}", domain);
    snprintf(r->data_json, sizeof(r->data_json), "{\"committed\":%s}",
             cell ? "true" : "false");
    r->success = (cell != NULL);
    return cell ? 0 : -1;
}

static int tool_epoch_advance(const char *args, TB_ToolResult *r, void *ud) {
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    int steps = 1;
    const char *p = strstr(args, "\"steps\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' ')p++; steps=(int)strtol(p,NULL,10); }
    if (steps < 1 || steps > 64) steps = 1;
    int32_t new_epoch = tb_tree_epoch_advance(tree, steps);
    snprintf(r->data_json, sizeof(r->data_json), "{\"epoch\":%d}", (int)new_epoch);
    r->success = 1;
    return 0;
}

static int tool_state_describe(const char *args, TB_ToolResult *r, void *ud) {
    (void)args;
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    tb_tree_describe(tree, r->data_json, sizeof(r->data_json));
    r->success = 1;
    return 0;
}

static int tool_ledger_verify(const char *args, TB_ToolResult *r, void *ud) {
    (void)args;
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    int broken = -1;
    int ok = tb_erl_verify_chain(tree->ledger, &broken);
    snprintf(r->data_json, sizeof(r->data_json),
             "{\"chain_valid\":%s,\"broken_at\":%d,\"n_entries\":%d}",
             ok?"true":"false", broken, tree->ledger->n_entries);
    r->success = 1;
    return 0;
}

static int tool_ledger_summary(const char *args, TB_ToolResult *r, void *ud) {
    (void)args;
    TB_CognitionTree *tree = (TB_CognitionTree*)ud;
    TB_ERLLedger *L = tree->ledger;
    char *wp = r->data_json;
    int rem = (int)sizeof(r->data_json);
    int n = snprintf(wp, rem, "{\"n_entries\":%d,\"last\":", L->n_entries);
    wp += n; rem -= n;
    if (L->n_entries > 0) {
        TB_ERLEntry *last = &L->entries[L->n_entries - 1];
        n = snprintf(wp, rem, "{\"seq\":%d,\"type\":%d,\"data\":%s}}",
                     last->seq, (int)last->type, last->data);
    } else {
        n = snprintf(wp, rem, "null}");
    }
    (void)n;
    r->success = 1;
    return 0;
}

static int tool_lattice_describe(const char *args, TB_ToolResult *r, void *ud) {
    (void)args;
    TB_PhiLattice *lat = (TB_PhiLattice*)ud;
    snprintf(r->data_json, sizeof(r->data_json),
             "{\"epoch\":%d,\"n_slots\":%d,\"phase_var\":%.4f}",
             (int)lat->epoch, lat->n_slots, (float)lat->phase_var);
    r->success = 1;
    return 0;
}

static int tool_lattice_advance(const char *args, TB_ToolResult *r, void *ud) {
    TB_PhiLattice *lat = (TB_PhiLattice*)ud;
    int steps = 1;
    const char *p = strstr(args, "\"steps\"");
    if (p) { p = strchr(p,':'); if(p)p++; while(*p==' ')p++; steps=(int)strtol(p,NULL,10); }
    if (steps < 1 || steps > 64) steps = 1;
    tb_lattice_advance(lat, steps);
    snprintf(r->data_json, sizeof(r->data_json), "{\"epoch\":%d}", (int)lat->epoch);
    r->success = 1;
    return 0;
}

/* ============================================================================
 * SECTION 3: Tool registry
 * ============================================================================ */

TB_ToolRegistry* tb_tool_registry_create(TB_CognitionTree *tree) {
    TB_ToolRegistry *reg = calloc(1, sizeof(*reg));
    if (!reg) return NULL;
    reg->tree = tree;

    /* ── Register 19 built-in tools ─────────────────────────────────────── */
    TB_PhiLattice *lat = tree->lattice;

    static const struct { const char *n; const char *d; TB_ToolFn fn; int tree_arg; int lat_arg; } BUILTINS[] = {
        {"shell_exec",       "Execute a shell command (popen). Args: {\"cmd\":\"...\"}",   tool_shell_exec,    0, 0},
        {"read_file",        "Read a file. Args: {\"path\":\"...\"}",                      tool_read_file,     0, 0},
        {"write_file",       "Write a file. Args: {\"path\":\"...\",\"content\":\"...\"}",  tool_write_file,    0, 0},
        {"list_dir",         "List directory entries. Args: {\"path\":\"...\"}",           tool_list_dir,      0, 0},
        {"file_exists",      "Check if path exists. Args: {\"path\":\"...\"}",             tool_file_exists,   0, 0},
        {"make_dir",         "Create directory. Args: {\"path\":\"...\"}",                 tool_make_dir,      0, 0},
        {"memory_set",       "Store key/value in tree session memory.",                    tool_memory_set,    1, 0},
        {"memory_get",       "Retrieve key from tree session memory.",                     tool_memory_get,    1, 0},
        {"notes_write",      "Write a notes file in tree persist_dir.",                    tool_notes_write,   1, 0},
        {"notes_read",       "Read a notes file from tree persist_dir.",                   tool_notes_read,    1, 0},
        {"branch_create",    "Create a new cognition branch. Args: {\"from\":0}",          tool_branch_create, 1, 0},
        {"branch_merge",     "Merge two branches. Args: {\"src\":1,\"dst\":0}",            tool_branch_merge,  1, 0},
        {"cell_commit",      "Commit a cell to a branch.",                                 tool_cell_commit,   1, 0},
        {"epoch_advance",    "Advance the lattice epoch. Args: {\"steps\":1}",             tool_epoch_advance, 1, 0},
        {"state_describe",   "Return JSON description of the cognition tree.",             tool_state_describe,1, 0},
        {"ledger_verify",    "Verify the ERL chain integrity.",                            tool_ledger_verify, 1, 0},
        {"ledger_summary",   "Return a summary of the ERL ledger.",                        tool_ledger_summary,1, 0},
        {"lattice_describe", "Return JSON description of the phi-lattice.",                tool_lattice_describe,0,1},
        {"lattice_advance",  "Step the phi-lattice RK4. Args: {\"steps\":1}",             tool_lattice_advance, 0,1},
        {NULL, NULL, NULL, 0, 0}
    };

    for (int i = 0; BUILTINS[i].n; i++) {
        void *ud = BUILTINS[i].tree_arg ? (void*)tree :
                   BUILTINS[i].lat_arg  ? (void*)lat   : NULL;
        tb_tool_register(reg, BUILTINS[i].n, BUILTINS[i].d, BUILTINS[i].fn, ud);
    }

    return reg;
}

void tb_tool_registry_destroy(TB_ToolRegistry *reg) { free(reg); }

int tb_tool_register(TB_ToolRegistry *reg, const char *name,
                      const char *desc, TB_ToolFn fn, void *userdata) {
    if (!reg || reg->n_tools >= TB_TOOL_REGISTRY_MAX) return -1;
    int idx = reg->n_tools++;
    snprintf(reg->tools[idx].name,        TB_TOOL_NAME_LEN, "%s", name ? name : "");
    snprintf(reg->tools[idx].description, TB_TOOL_DESC_LEN, "%s", desc ? desc : "");
    reg->tools[idx].fn       = fn;
    reg->tools[idx].userdata = userdata;
    return idx;
}

int tb_tool_call(TB_ToolRegistry *reg, const char *name,
                  const char *args_json, TB_ToolResult *res) {
    memset(res, 0, sizeof(*res));
    for (int i = 0; i < reg->n_tools; i++) {
        if (strcmp(reg->tools[i].name, name) == 0) {
            if (!reg->tools[i].fn) {
                snprintf(res->error, sizeof(res->error), "tool has no fn");
                return -1;
            }
            return reg->tools[i].fn(args_json ? args_json : "{}", res,
                                      reg->tools[i].userdata);
        }
    }
    snprintf(res->error, sizeof(res->error), "unknown tool: %s", name);
    return -1;
}

/* ============================================================================
 * SECTION 4: Task compiler — keyword → TB_Graph (linear chain)
 * ============================================================================ */

TB_Graph* tb_task_compile(const char *task, TB_PhiLattice *lat) {
    if (!task) return NULL;

    /* Keyword tables */
    static const char *SHELL_W[] = {
        "run","exec","execute","bash","sh","python","pip","npm","node","make",
        "git","docker","install","start","stop","kill","grep","find","cat",
        "curl","wget","ssh","rsync","cp","mv","rm","chmod","chown",NULL};
    static const char *FILE_W[] = {
        "read","open","load","parse","view","show","inspect","cat","head","tail",
        "less","more","display","print","content","file","import",NULL};
    static const char *CODE_W[] = {
        "analyze","analyse","fix","debug","optimize","refactor","review","audit",
        "scan","check","lint","compile","test","verify","validate",NULL};
    static const char *STORE_W[] = {
        "write","save","update","create","modify","delete","remove","generate",
        "build","output","export","produce","store","put","set",NULL};
    static const char *RECALL_W[] = {
        "remember","recall","find","search","look","memory","history","previous",
        "last","latest","context","retrieve","fetch","get","lookup",NULL};
    static const char *TRANSFORM_W[] = {
        "transform","convert","format","process","extract","merge","sort","filter",
        "map","reduce","aggregate","parse","tokenize","encode","decode",NULL};
    static const char *RESPOND_W[] = {
        "respond","answer","tell","say","explain","summarize","describe",
        "report","return","output","show","print","display","reply",NULL};
    static const char *BROWSE_W[] = {
        "browse","http","https","url","web","fetch","download","scrape",
        "request","api","endpoint","post","get",NULL};

    /* Lowercase copy of task for matching. */
    char task_lo[512];
    int tl = 0;
    for (; task[tl] && tl < 511; tl++)
        task_lo[tl] = (task[tl]>='A'&&task[tl]<='Z') ? task[tl]+32 : task[tl];
    task_lo[tl] = '\0';

    TB_PassType types[16]; int n_types = 0;

    char *tok = strtok(task_lo, " \t\n.,!?;:");
    while (tok && n_types < 12) {
        int matched = 0;
        #define MATCH(WL, PT) if(!matched) for(int _i=0;WL[_i]&&!matched;_i++) \
            if(strcmp(tok,WL[_i])==0){types[n_types++]=(PT);matched=1;}
        MATCH(SHELL_W,     TB_PASS_SHELL)
        MATCH(FILE_W,      TB_PASS_FETCH)
        MATCH(CODE_W,      TB_PASS_CODE)
        MATCH(STORE_W,     TB_PASS_STORE)
        MATCH(RECALL_W,    TB_PASS_RECALL)
        MATCH(TRANSFORM_W, TB_PASS_TRANSFORM)
        MATCH(RESPOND_W,   TB_PASS_RESPOND)
        MATCH(BROWSE_W,    TB_PASS_BROWSE)
        #undef MATCH
        tok = strtok(NULL, " \t\n.,!?;:");
    }

    /* Default if no keywords matched. */
    if (n_types == 0) {
        types[0] = TB_PASS_RECALL;
        types[1] = TB_PASS_TRANSFORM;
        types[2] = TB_PASS_RESPOND;
        n_types = 3;
    }

    /* Deduplicate (preserve first occurrence). */
    TB_PassType uniq[16]; int n_uniq = 0;
    for (int i = 0; i < n_types; i++) {
        int dup = 0;
        for (int j = 0; j < n_uniq; j++) if (uniq[j] == types[i]) { dup=1; break; }
        if (!dup) uniq[n_uniq++] = types[i];
    }

    /* Ensure RESPOND is last. */
    int has_respond = 0;
    for (int i = 0; i < n_uniq; i++) if (uniq[i] == TB_PASS_RESPOND) has_respond = 1;
    if (!has_respond && n_uniq < 16) uniq[n_uniq++] = TB_PASS_RESPOND;

    /* Build linear graph. */
    TB_Graph *g = tb_graph_create("unfold", lat);
    if (!g) return NULL;
    TB_Node *prev = NULL;
    for (int i = 0; i < n_uniq; i++) {
        char nname[32];
        snprintf(nname, sizeof(nname), "%s_%d", tb_pass_name(uniq[i]), i);
        TB_Node *n = tb_graph_add_node(g, uniq[i], nname, prev ? &prev : NULL, prev?1:0);
        prev = n;
    }
    tb_graph_topo_sort(g);
    return g;
}

/* ============================================================================
 * SECTION 5: Graph execution (op bindings per pass type)
 * ============================================================================ */

typedef struct {
    TB_ToolRegistry  *reg;
    TB_CognitionTree *tree;
    int               branch_id;
    const char       *task;
    TB_ExecStore     *store;
} ExecCtx;

static void op_dispatch(TB_Node *n, ExecCtx *ctx) {
    TB_ExecStore *s   = ctx->store;
    const char   *task = store_get(s, "_task");
    if (!task) task = ctx->task ? ctx->task : "";

    switch (n->pass_type) {

    case TB_PASS_RECALL: {
        /* Try to recall a value matching a trimmed-task key. */
        char key[64]; int kl = 0;
        for (int i = 0; task[i] && kl < 63; i++)
            if (task[i] != ' ') key[kl++] = task[i];
        key[kl] = '\0';
        const char *v = tb_tree_memory_get(ctx->tree, key);
        if (v) store_set(s, "recalled", v);
        break;
    }

    case TB_PASS_SHELL: {
        TB_ToolResult r = {0};
        char args[640];
        /* Escape quotes in task to make minimal valid JSON. */
        char escaped[512]; int ei = 0;
        for (int i = 0; task[i] && ei < 510; i++) {
            if (task[i] == '"') escaped[ei++] = '\\';
            escaped[ei++] = task[i];
        }
        escaped[ei] = '\0';
        snprintf(args, sizeof(args), "{\"cmd\":\"%s\"}", escaped);
        if (tb_tool_call(ctx->reg, "shell_exec", args, &r) == 0)
            store_set(s, "shell_result", r.data_json);
        else
            store_set(s, "shell_error", r.error);
        break;
    }

    case TB_PASS_FETCH: {
        /* Try to find a file-like token in the task and read it. */
        char last_path[256] = {0};
        char tmp[512]; snprintf(tmp, sizeof(tmp), "%s", task);
        char *tok2 = strtok(tmp, " \t");
        while (tok2) {
            if (strchr(tok2, '/') || strchr(tok2, '.')) {
                snprintf(last_path, sizeof(last_path), "%s", tok2);
            }
            tok2 = strtok(NULL, " \t");
        }
        if (last_path[0]) {
            TB_ToolResult r = {0};
            char args[320];
            snprintf(args, sizeof(args), "{\"path\":\"%s\"}", last_path);
            if (tb_tool_call(ctx->reg, "read_file", args, &r) == 0)
                store_set(s, "file_content", r.data_json);
        }
        break;
    }

    case TB_PASS_TRANSFORM: {
        /* Aggregate all available results into a single JSON object. */
        char buf[2048]; int bw = 0;
        bw += snprintf(buf + bw, sizeof(buf) - bw, "{");
        int first = 1;
        const char *akeys[] = {"shell_result","file_content","recalled",
                                "browse_result", NULL};
        for (int i = 0; akeys[i]; i++) {
            const char *v = store_get(s, akeys[i]);
            if (!v) continue;
            if (!first) bw += snprintf(buf+bw, sizeof(buf)-bw, ",");
            /* Escape value for JSON string embedding (crude). */
            bw += snprintf(buf+bw, sizeof(buf)-bw, "\"%s\":\"%.*s\"",
                           akeys[i], 200, v);
            first = 0;
        }
        bw += snprintf(buf+bw, sizeof(buf)-bw, "}");
        store_set(s, "transform_result", buf);
        break;
    }

    case TB_PASS_STORE: {
        /* Persist transform_result or recalled into tree memory. */
        char key[64]; int kl = 0;
        for (int i = 0; task[i] && kl < 63; i++)
            if (task[i] != ' ') key[kl++] = task[i];
        key[kl] = '\0';
        const char *v = store_get(s, "transform_result");
        if (!v) v = store_get(s, "recalled");
        if (v && key[0]) tb_tree_memory_set(ctx->tree, key, v);
        break;
    }

    case TB_PASS_RESPOND: {
        const char *v = store_get(s, "transform_result");
        if (!v) v = store_get(s, "shell_result");
        if (!v) v = store_get(s, "file_content");
        if (!v) v = store_get(s, "recalled");
        store_set(s, "response", v ? v : "{}");
        break;
    }

    case TB_PASS_CODE: {
        /* Code analysis: aggregate available data, build structured result. */
        const char *src = store_get(s, "file_content");
        if (!src) src = store_get(s, "recalled");
        if (!src) src = store_get(s, "shell_result");
        if (src) {
            /* Count lines, detect language, emit structured JSON */
            int lines = 0;
            for (const char *p = src; *p; p++) if (*p == '\n') lines++;
            const char *lang = "c";
            if (strstr(src, "import ") && strstr(src, "def "))    lang = "python";
            else if (strstr(src, "#include"))                      lang = "c";
            else if (strstr(src, "function ") || strstr(src,"{"))  lang = "js";
            char buf[2048];
            snprintf(buf, sizeof(buf),
                     "{\"lang\":\"%s\",\"lines\":%d,\"analysis\":\"%.*s\"}",
                     lang, lines, 300, src);
            store_set(s, "transform_result", buf);
            store_set(s, "code_analysis", buf);
        }
        break;
    }

    case TB_PASS_BROWSE: {
        /* For now: try shell curl. */
        TB_ToolResult r = {0};
        char args[640];
        /* Extract URL from task (very crude: find http token). */
        char url[256] = {0};
        const char *up = strstr(task, "http");
        if (up) {
            int ui = 0;
            while (up[ui] && up[ui] != ' ' && up[ui] != '"' && ui < 255) {
                url[ui] = up[ui];
                ui++;
            }
            url[ui] = '\0';
        }
        if (url[0]) {
            snprintf(args, sizeof(args), "{\"cmd\":\"curl -s --max-time 5 '%s'\"}", url);
            if (tb_tool_call(ctx->reg, "shell_exec", args, &r) == 0)
                store_set(s, "browse_result", r.data_json);
        }
        break;
    }

    default:
        break;
    }
}

int tb_graph_execute(TB_Graph *g, TB_ToolRegistry *reg,
                     TB_CognitionTree *tree, int branch_id,
                     const char *task,
                     char store_keys[][128], char store_vals[][2048],
                     int store_cap) {
    TB_ExecStore local_store;
    memset(&local_store, 0, sizeof(local_store));

    /* Seed the store from the caller-provided initial entries (if any). */
    int init_n = 0;
    if (store_keys && store_vals) {
        for (int i = 0; i < store_cap; i++) {
            if (!store_keys[i][0]) break;
            store_set(&local_store, store_keys[i], store_vals[i]);
            init_n++;
        }
    }
    store_set(&local_store, "_task", task ? task : "");

    ExecCtx ctx = { reg, tree, branch_id, task, &local_store };

    tb_graph_topo_sort(g);
    for (int i = 0; i < g->n_topo; i++) {
        TB_Node *n = g->topo_order[i];
        if (!n || n->fused_into) continue;
        op_dispatch(n, &ctx);
    }

    /* Copy results back to caller store if provided. */
    if (store_keys && store_vals) {
        int written = 0;
        for (int i = 0; i < local_store.n && written < store_cap; i++) {
            snprintf(store_keys[written], 128,  "%s", local_store.keys[i]);
            snprintf(store_vals[written], 2048, "%s", local_store.vals[i]);
            written++;
        }
        return written;
    }
    return local_store.n;
}

/* ============================================================================
 * SECTION 6: Unfold engine
 * ============================================================================ */

TB_UnfoldEngine* tb_unfold_engine_create(TB_CognitionTree *tree,
                                          TB_ToolRegistry *reg,
                                          TB_PhiLattice *lat) {
    TB_UnfoldEngine *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->tree     = tree;
    e->registry = reg;
    e->lattice  = lat;
    return e;
}

void tb_unfold_engine_destroy(TB_UnfoldEngine *e) { free(e); }

TB_UnfoldResult tb_unfold(TB_UnfoldEngine *e, const char *task, int branch_id) {
    TB_UnfoldResult r;
    memset(&r, 0, sizeof(r));
    snprintf(r.task, sizeof(r.task), "%s", task ? task : "");
    r.branch_id = branch_id;

    if (!e || !task || !task[0]) {
        snprintf(r.error, sizeof(r.error), "invalid engine or empty task");
        return r;
    }

    int n0 = e->tree->ledger->n_entries;

    /* ERL: task start */
    char erl[TB_ERL_DATA_LEN];
    snprintf(erl, sizeof(erl), "{\"task\":\"%.*s\",\"branch\":%d}", 180, task, branch_id);
    tb_erl_append(e->tree->ledger, TB_ERL_TASK_START, branch_id, erl);

    double t0 = tb_orch_wall_ms();

    /* Compile. */
    TB_Graph *g = tb_task_compile(task, e->lattice);
    if (!g) {
        snprintf(r.error, sizeof(r.error), "task_compile returned NULL");
        return r;
    }
    tb_graph_fuse(g);
    r.graph_nodes = g->n_nodes;

    /* Collect pass-sequence names from topo order. */
    tb_graph_topo_sort(g);
    for (int i = 0; i < g->n_topo && r.n_passes < TB_UNFOLD_PASS_MAX; i++) {
        TB_Node *n = g->topo_order[i];
        if (!n || n->fused_into) continue;
        snprintf(r.pass_sequence[r.n_passes++], 32, "%s",
                 tb_pass_name(n->pass_type));
    }

    /* Execute. */
    char store_keys[TB_STORE_CAP][128];
    char store_vals[TB_STORE_CAP][2048];
    memset(store_keys, 0, sizeof(store_keys));
    memset(store_vals, 0, sizeof(store_vals));

    int n_written = tb_graph_execute(g, e->registry, e->tree, branch_id,
                                      task, store_keys, store_vals, TB_STORE_CAP);
    tb_graph_destroy(g);

    /* Pull response from store. */
    const char *resp = NULL;
    for (int i = 0; i < n_written; i++) {
        if (strcmp(store_keys[i], "response") == 0) { resp = store_vals[i]; break; }
    }
    if (!resp) {
        for (int i = 0; i < n_written; i++) {
            if (strcmp(store_keys[i], "transform_result") == 0) { resp = store_vals[i]; break; }
        }
    }
    snprintf(r.result_json, sizeof(r.result_json), "%s",
             (resp && resp[0]) ? resp : "{}");

    /* Commit result to cognition tree. */
    tb_tree_cell_commit(e->tree, branch_id, r.result_json, "unfold_result");

    /* ERL: task complete */
    r.exec_ms = tb_orch_wall_ms() - t0;
    snprintf(erl, sizeof(erl),
             "{\"task\":\"%.*s\",\"nodes\":%d,\"ms\":%.1f,\"ok\":true}",
             100, task, r.graph_nodes, r.exec_ms);
    tb_erl_append(e->tree->ledger, TB_ERL_TASK_COMPLETE, branch_id, erl);

    r.erl_entries_added = e->tree->ledger->n_entries - n0;
    r.success = 1;
    return r;
}

/* ============================================================================
 * SECTION 7: Agent fabric
 * ============================================================================ */

TB_AgentFabric* tb_agent_fabric_create(TB_UnfoldEngine *engine) {
    TB_AgentFabric *fab = calloc(1, sizeof(*fab));
    if (!fab) return NULL;
    fab->engine = engine;
    fab->tree   = engine->tree;
    return fab;
}

void tb_agent_fabric_destroy(TB_AgentFabric *fab) { free(fab); }

int tb_agent_delegate(TB_AgentFabric *fab, const char *task, int from_branch) {
    if (!fab || !task) return -1;

    /* Create a new branch for the sub-agent. */
    int sub_branch = tb_tree_branch_create(fab->tree, from_branch);
    if (sub_branch < 0) return -1;

    /* Record delegation in ERL. */
    char erl[TB_ERL_DATA_LEN];
    snprintf(erl, sizeof(erl),
             "{\"delegate\":true,\"from\":%d,\"to\":%d,\"task\":\"%.*s\"}",
             from_branch, sub_branch, 180, task);
    tb_erl_append(fab->tree->ledger, TB_ERL_AGENT_DELEGATE, from_branch, erl);

    /* Unfold the sub-task on the new branch. */
    TB_UnfoldResult r = tb_unfold(fab->engine, task, sub_branch);
    if (!r.success) return -1;

    return sub_branch;
}

/* ============================================================================
 * SECTION 8: HTTP extension handler (/unfold /tools /state /ledger /sse)
 * ============================================================================ */

static void tb_orch_http_resp(int fd, int code, const char *ctype,
                               const char *body) {
    const char *reason = code == 200 ? "OK"
                       : code == 404 ? "Not Found"
                       : code == 400 ? "Bad Request"
                       : "Internal Server Error";
    char hdr[512];
    int hlen = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        code, reason, ctype, strlen(body));
    write(fd, hdr, hlen);
    write(fd, body, strlen(body));
}

int tb_orch_http_handle(int fd, const char *path, const char *method,
                         const char *body, void *ctx) {
    TB_UnfoldEngine *e = (TB_UnfoldEngine*)ctx;

    /* ── GET /tools ───────────────────────────────────────────────────── */
    if (strcmp(path, "/tools") == 0) {
        char resp[4096];
        int rw = snprintf(resp, sizeof(resp), "{\"tools\":[");
        for (int i = 0; i < e->registry->n_tools; i++) {
            if (i > 0) rw += snprintf(resp+rw, sizeof(resp)-rw, ",");
            rw += snprintf(resp+rw, sizeof(resp)-rw,
                           "{\"name\":\"%s\",\"description\":\"%s\"}",
                           e->registry->tools[i].name,
                           e->registry->tools[i].description);
        }
        rw += snprintf(resp+rw, sizeof(resp)-rw, "]}");
        (void)rw;
        tb_orch_http_resp(fd, 200, "application/json", resp);
        return 0;
    }

    /* ── GET /state ───────────────────────────────────────────────────── */
    if (strcmp(path, "/state") == 0) {
        char resp[2048];
        tb_tree_describe(e->tree, resp, sizeof(resp));
        tb_orch_http_resp(fd, 200, "application/json", resp);
        return 0;
    }

    /* ── GET /ledger ──────────────────────────────────────────────────── */
    if (strcmp(path, "/ledger") == 0) {
        TB_ERLLedger *L = e->tree->ledger;
        char resp[8192];
        int broken = -1;
        int chain_ok = tb_erl_verify_chain(L, &broken);
        int rw = snprintf(resp, sizeof(resp),
                          "{\"n_entries\":%d,\"chain_valid\":%s,"
                          "\"broken_at\":%d,\"entries\":[",
                          L->n_entries, chain_ok?"true":"false", broken);
        int start = L->n_entries > 20 ? L->n_entries - 20 : 0;
        for (int i = start; i < L->n_entries; i++) {
            if (i > start) rw += snprintf(resp+rw, sizeof(resp)-rw, ",");
            TB_ERLEntry *ev = &L->entries[i];
            rw += snprintf(resp+rw, sizeof(resp)-rw,
                           "{\"seq\":%d,\"type\":%d,\"branch\":%d,"
                           "\"epoch\":%d,\"data\":%s}",
                           ev->seq, (int)ev->type, ev->branch_id,
                           (int)ev->epoch, ev->data);
        }
        rw += snprintf(resp+rw, sizeof(resp)-rw, "]}");
        (void)rw;
        tb_orch_http_resp(fd, 200, "application/json", resp);
        return 0;
    }

    /* ── POST /unfold ─────────────────────────────────────────────────── */
    if (strcmp(path, "/unfold") == 0 && strcmp(method, "POST") == 0) {
        char task[512] = "analyze";
        int  branch_id = 0;
        const char *bp = body ? body : "{}";

        /* Extract task from JSON body. */
        const char *tp = strstr(bp, "\"task\"");
        if (tp) {
            tp = strchr(tp, ':'); if (tp) tp++;
            while (*tp == ' ' || *tp == '"') tp++;
            int n = 0;
            while (tp[n] && tp[n] != '"' && n < 511) n++;
            memcpy(task, tp, n); task[n] = '\0';
        }
        const char *brp = strstr(bp, "\"branch_id\"");
        if (brp) {
            brp = strchr(brp, ':'); if (brp) brp++;
            while (*brp == ' ') brp++;
            branch_id = (int)strtol(brp, NULL, 10);
        }

        TB_UnfoldResult r = tb_unfold(e, task, branch_id);

        char resp[4096];
        snprintf(resp, sizeof(resp),
                 "{\"success\":%s,\"task\":\"%s\",\"branch_id\":%d,"
                 "\"graph_nodes\":%d,\"exec_ms\":%.1f,\"erl_entries\":%d,"
                 "\"result\":%s%s}",
                 r.success ? "true" : "false",
                 r.task, r.branch_id, r.graph_nodes, r.exec_ms,
                 r.erl_entries_added,
                 r.result_json[0] ? r.result_json : "{}",
                 r.error[0] ? ",\"error\":\"" : "");
        if (r.error[0]) {
            /* Append error and closing quote+brace. */
            char *end = resp + strlen(resp);
            snprintf(end, sizeof(resp) - (end - resp), "%s\"}", r.error);
        }

        tb_orch_http_resp(fd, r.success ? 200 : 500, "application/json", resp);
        return 0;
    }

    /* ── GET /sse  (server-sent events — ERL tail, then close) ───────── */
    if (strcmp(path, "/sse") == 0) {
        const char *sse_hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        write(fd, sse_hdr, strlen(sse_hdr));
        TB_ERLLedger *L = e->tree->ledger;
        int start = L->n_entries > 5 ? L->n_entries - 5 : 0;
        for (int i = start; i < L->n_entries; i++) {
            TB_ERLEntry *ev = &L->entries[i];
            char evt[512];
            int el = snprintf(evt, sizeof(evt),
                "data:{\"seq\":%d,\"type\":%d,\"data\":%s}\n\n",
                ev->seq, (int)ev->type, ev->data);
            write(fd, evt, el);
        }
        return 0;
    }

    tb_orch_http_resp(fd, 404, "application/json",
                      "{\"error\":\"not found\"}");
    return 0;
}
