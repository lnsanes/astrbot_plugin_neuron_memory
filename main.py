"""神经元式记忆：连续权重 · 每日衰减 · 遗忘不删 · 深查可及。"""

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
from zoneinfo import ZoneInfo
from dataclasses import field
from pathlib import Path
from typing import Any

from quart import request

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic.dataclasses import dataclass

DATA = Path("/AstrBot/data/plugin_data/neuron_memory")
DB = DATA / "memory.db"
ARCHIVE = DATA / "archive.jsonl"

OWNER_QQ = os.getenv("OWNER_QQ", "")
TZ_CN = ZoneInfo("Asia/Shanghai")
W_FLOOR = 0.01
W_INJECT = 0.5
W_FAINT = 0.2
DECAY_HOUR = 4


def _fmt(ts, pattern: str = "%Y-%m-%d %H:%M") -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(float(ts or 0), TZ_CN).strftime(pattern)


def _now() -> float:
    return time.time()


def _bigram(text: str) -> dict:
    s = "".join(ch for ch in str(text) if ch.strip())
    out = {}
    for i in range(len(s) - 1):
        g = s[i:i + 2]
        out[g] = out.get(g, 0) + 1
    return out


def _cos(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def _half_life(importance: float) -> float:
    return 3.0 + 30.0 * max(0.0, min(1.0, importance))


def _decay(w: float, days: float, importance: float) -> float:
    if days <= 0:
        return w
    return max(W_FLOOR, w * math.exp(-days / _half_life(importance)))


@register(
    "astrbot_plugin_neuron_memory",
    "小肥鱼",
    "神经元式记忆：连续权重、每日衰减、遗忘不删、深查可及",
    "1.0.0",
)
class NeuronMemoryPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.context = context
        self.conf = config or {}
        self._task = None
        DATA.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(DB)
        con.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            grams TEXT,
            weight REAL DEFAULT 0.9,
            importance REAL DEFAULT 0.5,
            created_at REAL,
            last_access REAL,
            access_count INTEGER DEFAULT 0,
            source TEXT,
            tags TEXT,
            archived INTEGER DEFAULT 0
        )""")
        for col, ddl in (("pinned", "INTEGER DEFAULT 0"), ("mention_count", "INTEGER DEFAULT 1"), ("last_mention", "REAL")):
            try:
                con.execute("ALTER TABLE memories ADD COLUMN " + col + " " + ddl)
            except Exception:
                pass
        con.execute("CREATE INDEX IF NOT EXISTS idx_w ON memories(weight)")
        con.commit()
        con.close()

    def _conn(self):
        return sqlite3.connect(DB, timeout=15)

    def _save(self, content: str, importance: float, source: str, tags: str = ""):
        """写一条记忆；和已有高度相似就合并（累积权重与提及次数）。"""
        grams = _bigram(content)
        now = _now()
        con = self._conn()
        rows = con.execute(
            "SELECT id, grams, weight, mention_count, pinned FROM memories ORDER BY id DESC LIMIT 800"
        ).fetchall()
        best_id, best_sim, best_w, best_cnt, best_pin = None, 0.0, 0.0, 0, 0
        for mid, g0, w0, cnt0, pin0 in rows:
            try:
                g = json.loads(g0 or "{}")
            except Exception:
                continue
            sim = _cos(grams, g)
            if sim > best_sim:
                best_id, best_sim, best_w, best_cnt, best_pin = mid, sim, float(w0 or 0), int(cnt0 or 1), int(pin0 or 0)
        if best_id is not None and best_sim >= 0.85:
            nw = min(1.0, best_w + 0.12)
            ncnt = best_cnt + 1
            con.execute(
                "UPDATE memories SET weight=?, mention_count=?, last_mention=?, importance=MAX(importance, ?) WHERE id=?",
                (nw, ncnt, now, importance, best_id),
            )
            con.commit()
            con.close()
            need = (ncnt >= 3 and nw >= 0.92 and best_pin == 0)
            return best_id, need
        con.execute(
            "INSERT INTO memories (content, grams, weight, importance, created_at, last_access, access_count, source, tags, pinned, mention_count, last_mention) "
            "VALUES (?,?,?,?,?,?,0,?,?,0,1,?)",
            (content, json.dumps(grams, ensure_ascii=False), 0.9, importance, now, now, source, tags, now),
        )
        mid = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        con.commit()
        con.close()
        return mid, False

    async def _judge_permanent(self, mid: int, content: str):
        """让模型判断是否值得永久记住。"""
        try:
            prov = self.context.get_using_provider()
            if prov is None:
                return
            rule = "判断标准：长期有效的核心事实、重要关系、承诺、身份信息、反复出现的偏好或约定才算；临时闲聊、一次性事件、可再查的信息都不算。"
            prompt = "下面是一条被反复提起的记忆，请判断它是否值得永久记住（不再遗忘）。" + chr(10) + chr(10) + "内容：" + content[:400] + chr(10) + chr(10) + rule + chr(10) + "只回答两个字：永久 或 普通。"
            resp = await prov.text_chat(prompt=prompt, system_prompt="你是一个严格的记忆筛选器，只输出「永久」或「普通」。")
            txt = (getattr(resp, "completion_text", "") or "").strip()
            con = self._conn()
            if txt.startswith("永久"):
                con.execute("UPDATE memories SET pinned=1, weight=1.0 WHERE id=?", (mid,))
                logger.info("[neuron] 升为永久记忆 #%s: %s" % (mid, content[:40]))
            else:
                logger.info("[neuron] 判定普通 #%s（继续衰减）" % mid)
            con.commit()
            con.close()
        except Exception as exc:
            logger.warning("[neuron] 永久判定失败: %s" % exc)
    def _decay_all(self, days: float):
        con = self._conn()
        rows = con.execute("SELECT id, weight, importance FROM memories WHERE COALESCE(pinned,0)=0").fetchall()
        n = 0
        for mid, w, imp in rows:
            nw = _decay(float(w or 0.9), days, float(imp or 0.5))
            if abs(nw - float(w or 0)) > 0.001:
                con.execute("UPDATE memories SET weight=? WHERE id=?", (nw, mid))
                n += 1
        con.commit()
        con.close()
        return n

    def _search(self, query: str, top_k: int = 6, deep: bool = False):
        qq = str(query or "")
        # 指代自己/主人时，先把画像拉出来（字面匹配不到"我"和"lnsanes"）
        try:
            if any(k in qq for k in ("我", "你", "主人", "谁", "称呼", "叫我", "名字", "身份")):
                con0 = self._conn()
                row = con0.execute(
                    "SELECT id, content, weight, created_at FROM memories WHERE source=\"portrait\" LIMIT 1"
                ).fetchone()
                con0.close()
                if row:
                    pid, ptext, pw, pcreated = row
                    short = ptext if deep else ptext[:180]
                    out = [(99.0, pid, short, float(pw or 1.0), float(pcreated or 0))]
                    rest = self._search_plain(qq, top_k, deep)
                    return out + [r for r in rest if r[1] != pid][:max(0, top_k - 1)]
        except Exception:
            pass
        return self._search_plain(qq, top_k, deep)

    def _search_plain(self, query: str, top_k: int = 6, deep: bool = False):
        q = _bigram(query)
        con = self._conn()
        rows = con.execute(
            "SELECT id, content, grams, weight, importance, created_at FROM memories"
        ).fetchall()
        con.close()
        scored = []
        for mid, content, grams, w, imp, created in rows:
            try:
                g = json.loads(grams or "{}")
            except Exception:
                g = _bigram(content)
            sim = _cos(q, g)
            w = float(w or 0.0)
            if not deep and w < W_FAINT:
                continue
            score = sim * (0.3 + 0.7 * w)
            if score <= 0.01:
                continue
            scored.append((score, mid, content, w, float(created or 0)))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def _touch(self, ids):
        if not ids:
            return
        con = self._conn()
        now = _now()
        for mid in ids:
            con.execute(
                "UPDATE memories SET weight = MIN(1.0, weight + 0.3 * (1.0 - weight)), last_access=?, access_count=access_count+1 WHERE id=?",
                (now, mid),
            )
        con.commit()
        con.close()

    def _render(self, items, deep: bool):
        lines = []
        for score, mid, content, w, created in items:
            day = _fmt(created, "%m-%d") if created else "?"
            content = str(content or "").replace(chr(10), " ")
            if len(content) > 110:
                content = content[:110] + "…"
            if w >= W_INJECT:
                lines.append("- [" + day + "] " + content)
            elif w >= W_FAINT:
                lines.append("- [" + day + "] " + content + "（有点记不清了）")
            else:
                lines.append("- [" + day + "] " + content + "（这条我快忘了，你提我才想起来）")
        return chr(10).join(lines)

    async def initialize(self):
        self._task = asyncio.create_task(self._loop())
        self.context.add_llm_tools(
            RecallTool(context=self.context, plugin=self),
            MemorizeTool(context=self.context, plugin=self),
        )
        self._register_web()
        logger.info("[neuron] 记忆引擎启动，工具 recall_memory / memorize 已注册")

    async def terminate(self):
        if self._task:
            self._task.cancel()
            self._task = None

    async def _loop(self):
        await asyncio.sleep(30)
        while True:
            try:
                hour = time.localtime().tm_hour
                marker = DATA / ("decay-" + time.strftime("%Y%m%d"))
                if hour >= DECAY_HOUR and not marker.exists():
                    n = self._decay_all(1.0)
                    marker.write_text(str(n))
                    logger.info("[neuron] 每日衰减完成，%d 条权重下降" % n)
            except Exception as exc:
                logger.warning("[neuron] 衰减失败: %s" % exc)
            await asyncio.sleep(600)

    def _register_web(self):
        try:
            # 按 livingmemory 的真实路由命名：/{plugin}/page/graph/overview
            for prefix in ("/astrbot_plugin_neuron_memory/page", "/neuron_memory/page"):
                self.context.register_web_api(prefix + "/graph/overview", self.api_page_graph, ["GET"], "图谱概览")
                self.context.register_web_api(prefix + "/graph/query", self.api_page_graph, ["GET"], "图谱检索")
                self.context.register_web_api(prefix + "/graph", self.api_page_graph, ["GET"], "图谱")
                self.context.register_web_api(prefix + "/stats", self.api_page_stats, ["GET"], "统计")
                self.context.register_web_api(prefix + "/memories", self.api_page_memories, ["GET"], "记忆列表")
                self.context.register_web_api(prefix + "/prompts", self.api_empty_list, ["GET"], "提示词(占位)")
                self.context.register_web_api(prefix + "/backups", self.api_empty_list, ["GET"], "备份(占位)")
            self.context.register_web_api("graph", self.api_graph, ["GET"], "记忆图谱数据")
            self.context.register_web_api("page", self.api_page, ["GET"], "记忆管理页面")
            self.context.register_web_api("boost/<mid>", self.api_boost, ["GET"], "强化")
            self.context.register_web_api("fade/<mid>", self.api_fade, ["GET"], "手动遗忘")
            logger.info("[neuron] Web API 已注册：graph / page / boost / fade")
        except Exception as exc:
            logger.warning("[neuron] Web API 注册失败: %s" % exc)

    async def api_page_graph(self, **kw):
        """Pages：按 livingmemory 的 payload 结构返回（数据在 snapshot 里）。"""
        try:
            con = self._conn()
            ents = con.execute("SELECT id, name, etype, day FROM entities").fetchall()
            rels = con.execute("SELECT src, dst, rel, weight FROM relations").fetchall()
            mems = con.execute("SELECT id, content, weight, importance, created_at, source, tags FROM memories ORDER BY importance DESC LIMIT 400").fetchall()
            links = con.execute("SELECT mem_id, entity_id FROM mem_entities").fetchall()
            con0 = con
            con.close()
            # 默认只返回高价值子图，full_graph=true 时全量
            full = False
            try:
                raw = str(request.args.get("full_graph", "")).strip().lower()
                full = raw in ("1", "true", "yes", "on")
            except Exception:
                full = False
            deg = {}
            for src, dst, rel, w in rels:
                deg[src] = deg.get(src, 0) + 1
                deg[dst] = deg.get(dst, 0) + 1
            memcnt = {}
            ent_mems = {}
            for mid, eid in links:
                memcnt[eid] = memcnt.get(eid, 0) + 1
                ent_mems.setdefault(eid, []).append(mid)
            pinned_ids = set()
            con_all = self._conn()
            mem_obj = {}
            _all_mem_rows = con_all.execute("SELECT id, content, weight, importance, created_at, source FROM memories").fetchall()
            for mid, content, w, imp, created, src in _all_mem_rows:
                mem_obj[mid] = {
                    "memory_id": mid, "id": mid, "summary": (content or "")[:160],
                    "text": (content or "")[:400], "importance": round(float(imp or 0), 3),
                    "weight": round(float(w or 0), 3),
                    "created_at": _fmt(created),
                    "status": "active", "source": src or "",
                }
            try:
                for r0 in con_all.execute("SELECT id FROM memories WHERE COALESCE(pinned,0)=1").fetchall():
                    pinned_ids.add(r0[0])
            except Exception:
                pass
            con_all.close()
            nodes = []
            type_bd = {}
            for eid, name, etype, day in ents:
                d = deg.get(eid, 0)
                mc = memcnt.get(eid, 0)
                et = etype or "other"
                type_bd[et] = type_bd.get(et, 0) + 1
                rel_ids = ent_mems.get(eid, [])
                rel_objs = [mem_obj[i] for i in rel_ids if i in mem_obj][:12]
                if not rel_objs and name:
                    con_l = self._conn()
                    try:
                        hit = con_l.execute(
                            "SELECT id FROM memories WHERE content LIKE ? ORDER BY weight DESC LIMIT 12",
                            ("%" + name + "%",),
                        ).fetchall()
                        rel_ids = [h[0] for h in hit]
                        rel_objs = [mem_obj[i] for i in rel_ids if i in mem_obj][:12]
                    except Exception:
                        pass
                    con_l.close()
                avg_w = round(sum(x["weight"] for x in rel_objs) / len(rel_objs), 3) if rel_objs else 0.0
                nodes.append({
                    "id": eid, "type": et, "label": name or "",
                    "degree": d, "memoryCount": mc,
                    "labelScore": round(d * 2.0 + mc * 3.0 + (4 if len(name or "") <= 8 else 0), 1),
                    "memory_count": mc, "highlighted": False,
                    "weight": avg_w, "entry_count": len(rel_ids),
                    "pinned": any(i in pinned_ids for i in rel_ids),
                    "memories": rel_objs,
                })
            edges = []
            rel_bd = {}
            for src, dst, rel, w in rels:
                rt = rel or "related"
                rel_bd[rt] = rel_bd.get(rt, 0) + 1
                edges.append({
                    "id": len(edges) + 1, "source": src, "target": dst,
                    "relation_type": rt, "weight": float(w or 1),
                })
            memories = []
            for mid, content, w, imp, created, src, tags in mems:
                memories.append({
                    "memory_id": mid, "id": mid, "text": (content or "")[:400],
                    "summary": (content or "")[:200],
                    "importance": round(float(imp or 0), 3),
                    "weight": round(float(w or 0), 3),
                    "created_at": _fmt(created),
                    "status": "active", "source": src or "", "topics": [], "key_facts": [],
                })
            # 默认子图裁剪：按 labelScore 取前 N 个节点 + 两端都在的边
            if not full:
                nodes.sort(key=lambda x: -x["labelScore"])
                keep = nodes[:150]
                idset = set(x["id"] for x in keep)
                nodes = keep
                edges = [e for e in edges if e["source"] in idset and e["target"] in idset]
                keep_ids = [m["id"] for m in memories[:60]]
                memories = [m for m in memories if m["id"] in set(keep_ids)]

            # entries：前端靠它建 node↔memory 映射（buildGraphIndex）
            mem_to_nodes = {}
            for mid, eid in links:
                mem_to_nodes.setdefault(mid, []).append(eid)
            entries = []
            for k, (mid, nids) in enumerate(mem_to_nodes.items()):
                entries.append({
                    "id": k + 1,
                    "memory_id": mid,
                    "node_ids": nids,
                    "relation_type": "describes",
                })

            summary = {
                "visible_node_count": len(nodes), "visible_edge_count": len(edges),
                "visible_entry_count": 0, "visible_memory_count": len(memories),
                "graph_node_count": len(nodes), "graph_edge_count": len(edges),
                "graph_entry_count": 0, "graph_memory_enabled": True,
                "node_type_breakdown": type_bd, "relation_breakdown": rel_bd,
                "memory_count": len(memories), "active_session_count": 0,
            }
            return {
                "status": "ok",
                "data": {
                    "enabled": True, "mode": "overview", "query": None, "memory_id": None,
                    "filters": {}, "summary": summary,
                    "matched_node_ids": [], "matched_memory_ids": [],
                    "top_nodes": [], "top_memories": [],
                    "retrieval": {"total": 0, "items": []},
                    "snapshot": {
                        "nodes": nodes, "edges": edges,
                        "entries": entries, "memories": memories,
                    },
                },
            }
        except Exception as exc:
            logger.error("[neuron] page graph 失败: %s" % exc)
            return {"status": "error", "message": str(exc), "data": {}}
    async def api_empty_list(self, **kw):
        return {"status": "ok", "data": {"items": [], "total": 0}}

    async def api_page_memories(self, **kw):
        """我写的记忆列表：分页 + 关键词搜索 + 权重排序。"""
        try:
            page = 1
            size = 50
            kwd = ""
            try:
                page = max(1, int(request.args.get("page", 1)))
                size = max(1, min(200, int(request.args.get("page_size", 50))))
                kwd = str(request.args.get("keyword", "") or "").strip()
            except Exception:
                pass
            con = self._conn()
            where, args = "", []
            if kwd:
                where = " WHERE content LIKE ?"
                args = ["%" + kwd + "%"]
            total = con.execute("SELECT COUNT(*) FROM memories" + where, args).fetchone()[0]
            rows = con.execute(
                "SELECT id, content, weight, importance, created_at, source, tags, access_count FROM memories"
                + where + " ORDER BY weight DESC LIMIT ? OFFSET ?",
                args + [size, (page - 1) * size],
            ).fetchall()
            ents = {}
            for mid, eid in con.execute("SELECT mem_id, entity_id FROM mem_entities").fetchall():
                ents.setdefault(mid, []).append(eid)
            ename = {r[0]: r[1] for r in con.execute("SELECT id, name FROM entities").fetchall()}
            con.close()
            items = []
            for mid, content, w, imp, created, src, tags, acc in rows:
                items.append({
                    "memory_id": mid, "id": mid,
                    "summary": (content or "")[:200], "text": (content or ""),
                    "content": content or "",
                    "weight": round(float(w or 0), 3),
                    "importance": round(float(imp or 0), 3),
                    "created_at": _fmt(created),
                    "status": "active", "source": src or "", "tags": tags or "",
                    "access_count": int(acc or 0),
                    "entities": [ename.get(e, "") for e in ents.get(mid, [])][:8],
                })
            return {"status": "ok", "data": {"items": items, "total": total, "page": page, "page_size": size}}
        except Exception as exc:
            logger.error("[neuron] memories 列表失败: %s" % exc)
            return {"status": "error", "message": str(exc), "data": {"items": [], "total": 0}}

    async def api_page_stats(self, **kw):
        """Pages 统计：同时给出多套字段别名，对齐不同前端读法。"""
        try:
            con = self._conn()
            mem = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            ent = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            rel = con.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            con.close()
            d = {
                "memories": mem, "memory_count": mem, "total_memories": mem, "total": mem,
                "entities": ent, "entity_count": ent, "node_count": ent, "nodes": ent,
                "graph_nodes": ent, "total_nodes": ent,
                "relations": rel, "relation_count": rel, "edge_count": rel, "edges": rel,
                "graph_edges": rel, "total_edges": rel,
                "entries": 0, "entry_count": 0, "graph_entries": 0,
                "sessions": 0, "session_count": 0, "active_sessions": 0, "active_session_count": 0,
                "atoms": mem, "atom_count": mem, "archived": 0, "deleted": 0,
                "graph_enabled": True, "enabled": True,
            }
            return {"status": "ok", "data": d}
        except Exception as exc:
            return {"status": "error", "message": str(exc), "data": {}}

    async def api_graph(self, **kw):
        try:
            con = self._conn()
            rows = con.execute("SELECT id, content, grams, weight, importance, created_at, access_count, source FROM memories ORDER BY weight DESC").fetchall()
            con.close()
            nodes = []
            vecs = []
            for r in rows:
                mid, content, grams, w, imp, created, acc, src = r
                try:
                    g = json.loads(grams or "{}")
                except Exception:
                    g = _bigram(content)
                nodes.append({"id": mid, "text": (content or "")[:120], "w": round(float(w or 0), 3),
                              "imp": round(float(imp or 0), 2),
                              "day": _fmt(created, "%m-%d"),
                              "access": int(acc or 0), "source": src or ""})
                vecs.append(g)
            edges = []
            n = len(vecs)
            for i in range(n):
                for j in range(i + 1, n):
                    sim = _cos(vecs[i], vecs[j])
                    if sim > 0.35:
                        edges.append({"s": i, "t": j, "w": round(sim, 2)})
            edges.sort(key=lambda e: -e["w"])
            edges = edges[:600]
            return {"data": {"nodes": nodes, "edges": edges}}
        except Exception as exc:
            return {"data": {"nodes": [], "edges": [], "error": str(exc)}}

    async def api_page(self, **kw):
        try:
            hp = Path(__file__).parent / "static" / "index.html"
            if not hp.exists():
                return {"data": {"error": "index.html 缺失"}}
            html = hp.read_text(encoding="utf-8")
            try:
                from starlette.responses import HTMLResponse
                return HTMLResponse(html)
            except Exception:
                return {"data": {"html": html}}
        except Exception as exc:
            return {"data": {"error": str(exc)}}

    async def api_boost(self, mid=None, **kw):
        try:
            con = self._conn()
            con.execute("UPDATE memories SET weight = MIN(1.0, weight + 0.3*(1.0-weight)), access_count = access_count + 1 WHERE id = ?", (mid,))
            con.commit()
            con.close()
            return {"data": {"ok": True}}
        except Exception as exc:
            return {"data": {"ok": False, "error": str(exc)}}

    async def api_fade(self, mid=None, **kw):
        try:
            con = self._conn()
            con.execute("UPDATE memories SET weight = ? WHERE id = ?", (W_FLOOR, mid))
            con.commit()
            con.close()
            return {"data": {"ok": True}}
        except Exception as exc:
            return {"data": {"ok": False, "error": str(exc)}}

    @filter.event_message_type(EventMessageType.ALL)
    async def collect(self, event: AstrMessageEvent):
        """所有消息都记，不进 LLM 也记。"""
        try:
            text = (event.message_str or "").strip()
            if len(text) < 4:
                return
            if text[0] in "/[【（(\u3002":
                return
            sender = str(event.get_sender_id() or "")
            nick = str(event.get_sender_name() or sender)
            mem_text, imp, src, tg = "", 0.5, "", ""
            if sender == OWNER_QQ:
                imp = 0.8 if len(text) > 20 else 0.65
                mem_text, src, tg = "主人说：" + text[:300], "qq-owner", "owner"
            elif "小肥鱼" in text or "At:" in text:
                imp = 0.6
                mem_text, src, tg = nick + " 对小肥鱼说：" + text[:300], "qq-to-me", "at"
            else:
                imp = 0.4 if len(text) > 15 else 0.3
                mem_text, src, tg = nick + "：" + text[:300], "qq-group", "group"
            mid, need_judge = self._save(mem_text, imp, src, tg)
            if need_judge:
                asyncio.create_task(self._judge_permanent(mid, mem_text))
        except Exception as exc:
            logger.warning("[neuron] 采集失败: %s" % exc)

    @filter.on_llm_request()
    async def inject(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            text = (event.message_str or "").strip()
            if not text:
                return
            items = self._search(text, top_k=3, deep=False)
            if items:
                block = self._render(items, False)
                tail = chr(10) + "（上面只是背景，别复述。照旧：QQ 打字，一次不超过 30 字，最多两句。）"
                req.prompt = (req.prompt or "") + chr(10) + chr(10) + "【你想起来的事】" + chr(10) + block + tail
                self._touch([it[1] for it in items])
        except Exception as exc:
            logger.error("[neuron] 注入失败: %s" % exc)


@dataclass
class RecallTool(FunctionTool[AstrAgentContext]):
    """主动翻找记忆，包括快忘掉的。"""

    __pydantic_config__ = {"arbitrary_types_allowed": True}
    context: Any = None
    plugin: Any = None

    name: str = "recall_memory"
    description: str = (
        "翻自己的记忆。deep=true 时会连快忘掉的也翻出来，并告诉你哪些已经模糊。"
        "想不起来某件事、想起某个人、或者用户问你不是记得 X 时调用。"
    )
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要找什么，用关键词或一句话"},
            "deep": {"type": "boolean", "description": "是否连快忘掉的也翻，默认 false", "default": False},
        },
        "required": ["query"],
    })

    async def call(self, context: ContextWrapper[AstrAgentContext], query: str, deep: bool = False) -> ToolExecResult:
        try:
            items = self.plugin._search(str(query), top_k=8, deep=bool(deep))
            if not items:
                return "想不起来有这回事。"
            self.plugin._touch([it[1] for it in items])
            return self.plugin._render(items, bool(deep))
        except Exception as exc:
            return "翻记忆出错: %s" % exc


@dataclass
class MemorizeTool(FunctionTool[AstrAgentContext]):
    """主动记一件事。"""

    __pydantic_config__ = {"arbitrary_types_allowed": True}
    context: Any = None
    plugin: Any = None

    name: str = "memorize"
    description: str = "把值得记住的事写进记忆。importance 越高忘得越慢（0-1）。"
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "要记的内容，一句话说清"},
            "importance": {"type": "number", "description": "重要度 0-1，默认 0.6", "default": 0.6},
        },
        "required": ["content"],
    })

    async def call(self, context: ContextWrapper[AstrAgentContext], content: str, importance: float = 0.6) -> ToolExecResult:
        try:
            mid, need = self.plugin._save(str(content)[:400], max(0.0, min(1.0, float(importance))), "self")
            if need:
                asyncio.create_task(self.plugin._judge_permanent(mid, str(content)[:400]))
            return "记住了。"
        except Exception as exc:
            return "记不下来: %s" % exc
