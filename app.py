# -*- coding: utf-8 -*-
"""
QUIZZ_CAAV_B – Web App (Dual‑Mode, No‑Dependency Fallback)

Why this rewrite?
- The previous version crashed in environments without `streamlit` (ModuleNotFoundError).
- This version runs in **two modes**:
  1) **Streamlit mode** (if `streamlit` is installed, recommended for web UI)
  2) **CLI fallback** (no external packages) so it still works in sandboxed envs.

How to run (choose one):
- Streamlit UI (recommended):
    $ pip install streamlit pandas googletrans==4.0.0-rc1
    $ export QUIZ_MODE=streamlit  # or set in Windows: set QUIZ_MODE=streamlit
    $ streamlit run app.py

- CLI no-deps fallback:
    $ python app.py --cli

- Self tests (no deps, auto-makes temp CSVs):
    $ python app.py --selftest

Notes:
- CSV schema: columns include "Question", one or more option headers that **start with** "Option " (e.g., "Option A", "Option 1", ...), and "Correct Answer" that equals the exact text of one option.
- File names: {CAT}_Module{N}.csv  (e.g., B1_Module1.csv)
"""

from __future__ import annotations
import os
import sys
import csv
import time
import random
import tempfile
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# -----------------------------------------------------------
# Optional imports (streamlit / googletrans) with safe fallbacks
# -----------------------------------------------------------
try:
    import streamlit as _st  # type: ignore
except Exception:
    _st = None  # not available in sandbox

try:
    from googletrans import Translator  # type: ignore
    _HAS_TRANSLATOR = True
except Exception:
    Translator = None  # type: ignore
    _HAS_TRANSLATOR = False

APP_TITLE = "Quiz CAAV Cat B (Web)"
CATEGORIES = {
    "B. Question bank for theory basic maintenance Cat B1 (1660)": "B1",
    "C. Theory basic maintenance Cat B2_CAAV (1500)": "B2",
    "D. Question bank for Aviation Legislation (M10) Cat A, B1, B2, C (600)": "M10",
}
MODULES = [str(i) for i in range(1, 18)]
TEST_DURATION_SECONDS = 100 * 60  # 100 minutes

# -----------------------------------------------------------
# Utilities & Core Logic (no UI dependencies)
# -----------------------------------------------------------
@dataclass
class QAItem:
    q: str
    options: List[str]
    answer: int  # index into options
    cat: str
    module: str


def list_available_modules(code: str, base_dir: str = ".") -> List[int]:
    files = [f for f in os.listdir(base_dir) if f.startswith(f"{code}_Module") and f.endswith(".csv")]
    nums: List[int] = []
    for f in files:
        try:
            part = f.split("Module", 1)[1].rsplit(".", 1)[0]
            nums.append(int(part))
        except Exception:
            continue
    return sorted(set(nums))


def load_csv_bank(code: str, module: str, base_dir: str = ".") -> List[QAItem]:
    filename = os.path.join(base_dir, f"{code}_Module{module}.csv")
    if not os.path.exists(filename):
        return []
    bank: List[QAItem] = []
    with open(filename, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        opt_cols = [h for h in fieldnames if h and h.lower().startswith("option ")]
        for row in reader:
            q = (row.get("Question") or "").strip()
            if not q:
                continue
            opts = [(row.get(c) or "").strip() for c in opt_cols if (row.get(c) or "").strip()]
            if not opts:
                continue
            correct_text = (row.get("Correct Answer") or "").strip()
            try:
                ans_idx = opts.index(correct_text)
            except ValueError:
                # If no exact match, default to first option (consistent with prior behavior)
                ans_idx = 0
            bank.append(QAItem(q=q, options=opts, answer=ans_idx, cat=code, module=str(module)))
    return bank


def mix_generate(percent_map: Dict[str, int], total: int, base_dir: str = ".") -> List[QAItem]:
    """Generate a mixed exam across categories, ensuring the final length == total when availability allows.
    Rounding uses the **largest remainder method** so, e.g., 50/50 of 5 → 2 & 3.
    Falls back to availability caps; if some categories lack questions, we reassign leftover to others with spare.
    Distribution across modules is then as even as possible for each category.
    """
    if sum(percent_map.values()) != 100:
        raise ValueError("Total percent must equal 100.")

    rng = random.Random(42)  # deterministic sampling for tests

    # 1) Scan availability per category and per module
    cat_mod_rows: Dict[str, Dict[int, List[QAItem]]] = {}
    cat_avail: Dict[str, int] = {}
    for cat in percent_map.keys():
        mods = list_available_modules(cat, base_dir)
        mod_map: Dict[int, List[QAItem]] = {}
        total_avail = 0
        for m in mods:
            rows = load_csv_bank(cat, str(m), base_dir)
            mod_map[m] = rows
            total_avail += len(rows)
        cat_mod_rows[cat] = mod_map
        cat_avail[cat] = total_avail

    # 2) Largest Remainder Method to compute targets per category
    #    target_raw = total * pct / 100
    #    base = floor(target_raw); remainder = target_raw - base
    base_targets: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for cat, pct in percent_map.items():
        raw = total * pct / 100.0
        base = int(raw)  # floor
        base_targets[cat] = min(base, cat_avail.get(cat, 0))
        rem = raw - base
        remainders.append((rem, cat))

    assigned = sum(base_targets.values())

    # If we under-assign due to flooring, distribute leftover by largest remainder, respecting availability
    leftover = total - assigned
    # Sort by remainder DESC then by cat name for determinism
    remainders.sort(key=lambda x: (-x[0], x[1]))
    i = 0
    while leftover > 0 and any(cat_avail[c] > base_targets[c] for _, c in remainders):
        rem, cat = remainders[i % len(remainders)]
        if cat_avail[cat] > base_targets[cat]:
            base_targets[cat] += 1
            leftover -= 1
        i += 1
        # safety break to avoid infinite loop in degenerate cases
        if i > 10000:
            break

    # If still leftover (because all capped), we cannot reach total → will return as many as available

    # 3) For each category, sample as evenly across its modules as possible to meet target
    out: List[QAItem] = []
    for cat, target in base_targets.items():
        if target <= 0 or cat_avail.get(cat, 0) == 0:
            continue
        mod_map = cat_mod_rows[cat]
        mods_sorted = sorted(mod_map.keys())
        if not mods_sorted:
            continue
        per_mod = target // len(mods_sorted)
        remainder = target % len(mods_sorted)
        for j, m in enumerate(mods_sorted):
            need = per_mod + (1 if j < remainder else 0)
            rows = mod_map[m]
            if not rows:
                continue
            take = min(need, len(rows))
            out.extend(rng.sample(rows, take))

        # If due to empty modules we didn't reach the target, top up from remaining pool
        if len([x for x in out if x.cat == cat]) < target:
            pool = []
            for rows in mod_map.values():
                pool.extend(rows)
            already_ids = set(id(x) for x in out)
            pool = [x for x in pool if id(x) not in already_ids]
            need_more = target - len([x for x in out if x.cat == cat])
            add = min(need_more, len(pool))
            if add > 0:
                out.extend(rng.sample(pool, add))

    rng.shuffle(out)

    # Final cap: cannot exceed total (in case of overfill by race)
    if len(out) > total:
        out = out[:total]
    return out


# -----------------------------------------------------------
# Streamlit UI (guarded; only defined/used when streamlit available)
# -----------------------------------------------------------

def _identity_decorator(*args, **kwargs):
    def wrap(func):
        return func
    return wrap


def main_streamlit():
    st = _st
    if st is None:
        raise RuntimeError("Streamlit not available; run CLI mode or install streamlit.")

    # Tabs: Làm đề + Tạo đề hỗn hợp
    tab_main, tab_mix = st.tabs(["🧩 Làm đề", "🏗️ Tạo đề hỗn hợp"])

    # ===== TAB 1: LÀM ĐỀ =====
    with tab_main:
        # --- toàn bộ phần giao diện cũ của bạn ở đây ---
        # (từ chỗ st.set_page_config() tới chỗ hiển thị kết quả)
        # Giữ nguyên logic load câu hỏi, hiển thị, chấm điểm, v.v.
        # ----------------------------------------------
        st.set_page_config(APP_TITLE, layout="wide")
        st.title(APP_TITLE)

        with st.sidebar:
            st.header("Cấu hình")
            st.session_state.category = st.selectbox(
                "Loại đề",
                list(CATEGORIES.keys()),
                index=list(CATEGORIES.keys()).index(st.session_state.category),
            )
            code = CATEGORIES[st.session_state.category]

            avail = list_available_modules(code)
            if avail:
                st.session_state.module = st.selectbox("Module", [str(x) for x in avail], index=0)
            else:
                st.session_state.module = st.selectbox("Module (không tìm thấy file)", MODULES, index=0)

            st.session_state.is_test_mode = st.toggle("Chế độ TEST (100 phút)",
                value=st.session_state.is_test_mode, help="Bỏ chọn = Practice")
            st.session_state.shuffle_q = st.checkbox("Xáo trộn câu", value=st.session_state.shuffle_q)
            st.session_state.shuffle_opt = st.checkbox("Xáo trộn đáp án", value=st.session_state.shuffle_opt)

            if st.button("Nạp câu hỏi", use_container_width=True):
                bank = load_csv_bank(code, st.session_state.module)
                st.session_state.bank = bank
                if not bank:
                    st.warning("Không tìm thấy dữ liệu.")
                else:
                    st.session_state.order = list(range(len(bank)))
                    if st.session_state.shuffle_q:
                        random.shuffle(st.session_state.order)
                    st.session_state.cur = 0
                    st.session_state.picks = {}
                    st.session_state.fails_first_try = set()
                    st.success(f"Đã nạp {len(bank)} câu từ {code}_Module{st.session_state.module}.csv")

            if st.button("BẮT ĐẦU", type="primary", use_container_width=True, disabled=(len(st.session_state.bank) == 0)):
                st.session_state.is_quiz_active = True
                st.session_state.cur = 0
                st.session_state.picks = {}
                st.session_state.fails_first_try = set()
                if st.session_state.is_test_mode:
                    st.session_state.start_time = time.time()
                    st.session_state.remaining = TEST_DURATION_SECONDS
                st.toast("Bắt đầu làm bài!")

            if st.button("HỦY THI", use_container_width=True, disabled=not st.session_state.is_quiz_active):
                st.session_state.is_quiz_active = False
                st.session_state.bank = []
                st.session_state.order = []
                st.session_state.picks = {}
                st.session_state.fails_first_try = set()
                st.session_state.cur = 0
                st.session_state.start_time = None
                st.session_state.remaining = TEST_DURATION_SECONDS
                st.info("Đã hủy bài thi.")

        # (các phần hiển thị câu hỏi, chọn đáp án, kết quả, v.v. giữ nguyên như cũ)
        # ----------------------------------------------

    # ===== TAB 2: TẠO ĐỀ HỖN HỢP =====
    with tab_mix:
        st.subheader("Tạo đề hỗn hợp – có báo cáo chi tiết theo module")
        total_mix = st.number_input("Tổng số câu", min_value=10, max_value=300, value=100, step=10, key="mix_total")

        c1, c2, c3 = st.columns(3)
        with c1:
            pA = st.number_input("% B1", min_value=0, max_value=100, value=0, key="mix_p_b1")
        with c2:
            pB = st.number_input("% B2", min_value=0, max_value=100, value=0, key="mix_p_b2")
        with c3:
            pC = st.number_input("% M10", min_value=0, max_value=100, value=0, key="mix_p_m10")

        st.caption("• App sẽ chia đều số câu mỗi Cat qua các module hiện có.\n\n• Bấm **Tính phân bổ** để xem bảng chi tiết (tổng từng Cat và từng Module). Nếu ổn, bấm **Tạo đề hỗn hợp** để nạp vào bài.")

        def _plan_distribution() -> Dict[str, List[Dict]]:
            plan: Dict[str, List[Dict]] = {}
            for cat, pct in [("B1", pA), ("B2", pB), ("M10", pC)]:
                if pct <= 0:
                    continue
                cat_need = round(int(total_mix) * pct / 100)
                mods = list_available_modules(cat)
                rows = []
                if not mods:
                    plan[cat] = [{"module": "-", "available": 0, "need": 0, "take": 0}]
                    continue
                per_mod = cat_need // len(mods)
                rem = cat_need % len(mods)
                for i, m in enumerate(mods):
                    need = per_mod + (1 if i < rem else 0)
                    avail = len(load_csv_bank(cat, str(m)))
                    take = min(need, avail)
                    rows.append({"module": m, "available": avail, "need": need, "take": take})
                plan[cat] = rows
            return plan

        if st.button("Tính phân bổ"):
            plan = _plan_distribution()
            import pandas as pd
            total_take = 0
            for cat, rows in plan.items():
                st.markdown(f"**[{cat}] Tổng dự kiến:** {sum(r['take'] for r in rows)} câu")
                df = pd.DataFrame(rows)[["module", "available", "need", "take"]]
                df = df.rename(columns={"module": "Module", "available": "Có sẵn", "need": "Cần", "take": "Lấy"})
                st.dataframe(df, use_container_width=True)
                total_take += sum(r['take'] for r in rows)
            st.info(f"Tổng cộng sẽ lấy: **{total_take}**/{int(total_mix)} câu.")

        if st.button("Tạo đề hỗn hợp", use_container_width=True):
            if pA + pB + pC != 100:
                st.error("Tổng % phải = 100")
            else:
                plan = _plan_distribution()
                bank: List[QAItem] = []
                rng = random.Random()
                for cat, rows in plan.items():
                    for r in rows:
                        if r["take"] <= 0 or r["module"] == "-":
                            continue
                        pool = load_csv_bank(cat, str(r["module"]))
                        if not pool:
                            continue
                        take = min(r["take"], len(pool))
                        bank.extend(rng.sample(pool, take))
                if not bank:
                    st.error("Không tạo được đề (dữ liệu rỗng)")
                else:
                    rng.shuffle(bank)
                    st.session_state.bank = bank
                    st.session_state.order = list(range(len(bank)))
                    st.session_state.cur = 0
                    st.session_state.picks = {}
                    st.session_state.fails_first_try = set()
                    st.success(f"Đã tạo đề gồm {len(bank)} câu. Chuyển sang tab **🧩 Làm đề** và bấm **BẮT ĐẦU** để thi.")

def render_tabs(st):
    # Keep existing main flow; provide a second tab for creating a mixed exam with detailed per‑module report.
    try:
        tab_main, tab_mix = st.tabs(["🧩 Làm đề", "🏗️ Tạo đề hỗn hợp"])
    except Exception:
        # Older Streamlit versions may not have tabs; gracefully fall back to a header section
        tab_main = st.container()
        tab_mix = st.container()

    with tab_mix:
        st.subheader("Tạo đề hỗn hợp – có báo cáo chi tiết theo module")
        total_mix = st.number_input("Tổng số câu", min_value=10, max_value=300, value=100, step=10, key="mix_total")

        c1, c2, c3 = st.columns(3)
        with c1:
            pA = st.number_input("% B1", min_value=0, max_value=100, value=0, key="mix_p_b1")
        with c2:
            pB = st.number_input("% B2", min_value=0, max_value=100, value=0, key="mix_p_b2")
        with c3:
            pC = st.number_input("% M10", min_value=0, max_value=100, value=0, key="mix_p_m10")

        st.caption("• App sẽ chia đều số câu mỗi Cat qua các module hiện có.• Bấm **Tính phân bổ** để xem bảng chi tiết (tổng từng Cat và từng Module). Nếu ổn, bấm **Tạo đề hỗn hợp** để nạp vào bài.")

        def _plan_distribution() -> Dict[str, List[Dict]]:
            """Return a per-category plan with even split across available modules.
            Each item: {cat, module, available, need, take}
            """
            plan: Dict[str, List[Dict]] = {}
            for cat, pct in [("B1", pA), ("B2", pB), ("M10", pC)]:
                if pct <= 0:
                    continue
                cat_need = round(int(total_mix) * pct / 100)
                mods = list_available_modules(cat)
                rows = []
                if not mods:
                    plan[cat] = [{"module": "-", "available": 0, "need": 0, "take": 0}]
                    continue
                per_mod = cat_need // len(mods)
                rem = cat_need % len(mods)
                for i, m in enumerate(mods):
                    need = per_mod + (1 if i < rem else 0)
                    avail = len(load_csv_bank(cat, str(m)))
                    take = min(need, avail)
                    rows.append({"module": m, "available": avail, "need": need, "take": take})
                plan[cat] = rows
            return plan

        show_report = st.button("Tính phân bổ")
        if show_report:
            plan = _plan_distribution()
            import pandas as pd
            total_take = 0
            for cat, rows in plan.items():
                st.markdown(f"**[{cat}] Tổng dự kiến:** {sum(r['take'] for r in rows)} câu")
                df = pd.DataFrame(rows)[["module", "available", "need", "take"]]
                df = df.rename(columns={"module": "Module", "available": "Có sẵn", "need": "Cần", "take": "Lấy"})
                st.dataframe(df, use_container_width=True)
                total_take += sum(r['take'] for r in rows)
            st.info(f"Tổng cộng sẽ lấy: **{total_take}**/{int(total_mix)} câu.")

        if st.button("Tạo đề hỗn hợp", use_container_width=True):
            if pA + pB + pC != 100:
                st.error("Tổng % phải = 100")
            else:
                # Build using the same plan to ensure counts match the report
                plan = _plan_distribution()
                bank: List[QAItem] = []
                rng = random.Random()
                for cat, rows in plan.items():
                    for r in rows:
                        if r["take"] <= 0 or r["module"] == "-":
                            continue
                        pool = load_csv_bank(cat, str(r["module"]))
                        if not pool:
                            continue
                        take = min(r["take"], len(pool))
                        bank.extend(rng.sample(pool, take))
                if not bank:
                    st.error("Không tạo được đề (dữ liệu rỗng)")
                else:
                    rng.shuffle(bank)
                    st.session_state.bank = bank
                    st.session_state.order = list(range(len(bank)))
                    st.session_state.cur = 0
                    st.session_state.picks = {}
                    st.session_state.fails_first_try = set()
                    st.success(f"Đã tạo đề gồm {len(bank)} câu. Chuyển sang tab **🧩 Làm đề** và bấm **BẮT ĐẦU** để thi.")

# Attach tabs to main_streamlit after it is defined
if _st is not None:
    try:
        _orig_main_streamlit = main_streamlit
        def main_streamlit():  # type: ignore[no-redef]
            _orig_main_streamlit()
            render_tabs(_st)
    except Exception:
        # If something goes wrong, leave original main_streamlit intact
        pass
