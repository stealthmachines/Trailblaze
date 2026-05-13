/*
 * tb_orchestration.h — TRAILBLAZE Layer 3: Agent Orchestration
 *
 * TB_ToolRegistry    — 19 built-in tools + user-registered slots (max 32).
 * TB_UnfoldEngine    — task-string → TB_Graph → execute → ERL commit.
 * TB_AgentFabric     — multi-agent delegation via cognition branches.
 *
 * HTTP extension routes:
 *   POST /unfold  — run a task, return TB_UnfoldResult JSON
 *   GET  /tools   — list registered tools
 *   GET  /state   — tb_tree_describe() JSON
 *   GET  /ledger  — last N ERL entries + chain-valid flag
 *   GET  /sse     — server-sent events: recent ERL tail
 *
 * Ported from v0.1/orchestration.py — pure C11, POSIX popen/opendir.
 */

#pragma once
#ifndef TB_ORCHESTRATION_H
#define TB_ORCHESTRATION_H

#include <stdint.h>
#include <stddef.h>
#include "../layer0/tb_phi_lattice.h"
#include "../layer2/tb_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Tool system
 * ============================================================================ */

#define TB_TOOL_NAME_LEN   64
#define TB_TOOL_DESC_LEN   256
#define TB_TOOL_RESULT_LEN 2048
#define TB_TOOL_ERROR_LEN  256
#define TB_TOOL_REGISTRY_MAX 32

typedef struct {
    int  success;
    char data_json[TB_TOOL_RESULT_LEN];
    char error[TB_TOOL_ERROR_LEN];
} TB_ToolResult;

/* Tool function signature.  args_json is a JSON string from the caller.
 * Write result into *res.  Returns 0 on success, -1 on error. */
typedef int (*TB_ToolFn)(const char *args_json,
                         TB_ToolResult *res,
                         void *userdata);

typedef struct {
    char       name[TB_TOOL_NAME_LEN];
    char       description[TB_TOOL_DESC_LEN];
    TB_ToolFn  fn;
    void      *userdata;
} TB_Tool;

typedef struct {
    TB_Tool           tools[TB_TOOL_REGISTRY_MAX];
    int               n_tools;
    TB_CognitionTree *tree;       /* needed by tree-backed tools */
} TB_ToolRegistry;

/* ============================================================================
 * Unfold engine
 * ============================================================================ */

#define TB_UNFOLD_PASS_MAX 16

typedef struct {
    int    success;
    char   task[256];
    int    branch_id;
    char   pass_sequence[TB_UNFOLD_PASS_MAX][32];
    int    n_passes;
    int    graph_nodes;
    double exec_ms;
    int    erl_entries_added;
    char   result_json[TB_TOOL_RESULT_LEN];
    char   error[TB_TOOL_ERROR_LEN];
} TB_UnfoldResult;

typedef struct {
    TB_CognitionTree *tree;
    TB_ToolRegistry  *registry;
    TB_PhiLattice    *lattice;
} TB_UnfoldEngine;

/* ============================================================================
 * Agent fabric
 * ============================================================================ */

typedef struct {
    TB_UnfoldEngine  *engine;
    TB_CognitionTree *tree;
} TB_AgentFabric;

/* ============================================================================
 * API
 * ============================================================================ */

/* Registry lifecycle. */
TB_ToolRegistry* tb_tool_registry_create(TB_CognitionTree *tree);
void             tb_tool_registry_destroy(TB_ToolRegistry *reg);

/* Register a user tool (returns slot index or -1 if full). */
int  tb_tool_register(TB_ToolRegistry *reg, const char *name,
                       const char *desc, TB_ToolFn fn, void *userdata);

/* Call a tool by name. Returns 0 on success. */
int  tb_tool_call(TB_ToolRegistry *reg, const char *name,
                   const char *args_json, TB_ToolResult *res);

/* Unfold engine lifecycle. */
TB_UnfoldEngine* tb_unfold_engine_create(TB_CognitionTree *tree,
                                          TB_ToolRegistry *reg,
                                          TB_PhiLattice *lat);
void             tb_unfold_engine_destroy(TB_UnfoldEngine *e);

/* Compile a task string into a TB_Graph (caller frees with tb_graph_destroy). */
TB_Graph* tb_task_compile(const char *task, TB_PhiLattice *lat);

/* Execute a compiled graph against a key-value store.
 * store_keys / store_vals must be parallel arrays of capacity store_cap.
 * Returns n_entries written to store. */
int tb_graph_execute(TB_Graph *g, TB_ToolRegistry *reg,
                     TB_CognitionTree *tree, int branch_id,
                     const char *task,
                     char store_keys[][128], char store_vals[][2048],
                     int store_cap);

/* High-level: compile + execute + ERL + cell_commit. */
TB_UnfoldResult tb_unfold(TB_UnfoldEngine *e, const char *task, int branch_id);

/* Agent fabric. */
TB_AgentFabric* tb_agent_fabric_create(TB_UnfoldEngine *engine);
void            tb_agent_fabric_destroy(TB_AgentFabric *fab);

/* Delegate a sub-task to a new cognition branch.
 * Returns the branch_id of the spawned agent, or -1 on failure. */
int tb_agent_delegate(TB_AgentFabric *fab, const char *task, int from_branch);

/* ============================================================================
 * HTTP extension handler
 *
 * Register as cfg->ext_handler / cfg->ext_ctx in TB_ServeConfig.
 * Handles: POST /unfold  GET /tools  GET /state  GET /ledger  GET /sse
 * ============================================================================ */

int tb_orch_http_handle(int fd, const char *path, const char *method,
                         const char *body, void *ctx /* TB_UnfoldEngine* */);

#ifdef __cplusplus
}
#endif

#endif /* TB_ORCHESTRATION_H */
