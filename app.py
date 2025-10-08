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
    st = _st  # local alias
    if st is None:
        raise RuntimeError("Streamlit not available; run CLI mode or install streamlit.")

    # Use cache decorator if present, else no-op
    cache = getattr(st, "cache_data", _identity_decorator)

    @cache(show_spinner=False)
    def _list_mods(code: str) -> List[int]:
        return list_available_modules(code)

    @cache(show_spinner=True)
    def _load_bank(code: str, module: str) -> List[Dict]:
        return [i.__dict__ for i in load_csv_bank(code, module)]

    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # ---------- App State ----------
    if "bank" not in st.session_state:
        st.session_state.bank = []
    if "order" not in st.session_state:
        st.session_state.order = []
    if "cur" not in st.session_state:
        st.session_state.cur = 0
    if "picks" not in st.session_state:
        st.session_state.picks = {}
    if "fails_first_try" not in st.session_state:
        st.session_state.fails_first_try = set()
    if "is_quiz_active" not in st.session_state:
        st.session_state.is_quiz_active = False
    if "is_test_mode" not in st.session_state:
        st.session_state.is_test_mode = False
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "remaining" not in st.session_state:
        st.session_state.remaining = TEST_DURATION_SECONDS
    if "shuffle_q" not in st.session_state:
        st.session_state.shuffle_q = False
    if "shuffle_opt" not in st.session_state:
        st.session_state.shuffle_opt = False
    if "category" not in st.session_state:
        st.session_state.category = list(CATEGORIES.keys())[0]
    if "module" not in st.session_state:
        st.session_state.module = MODULES[0]

    with st.sidebar:
        st.header("Cấu hình")
        st.session_state.category = st.selectbox(
            "Loại đề", list(CATEGORIES.keys()), index=list(CATEGORIES.keys()).index(st.session_state.category)
        )
        code = CATEGORIES[st.session_state.category]

        avail = _list_mods(code)
        if avail:
            st.session_state.module = st.selectbox("Module", [str(x) for x in avail], index=0)
        else:
            st.session_state.module = st.selectbox("Module (không tìm thấy file)", MODULES, index=0)

        st.session_state.is_test_mode = st.toggle(
            "Chế độ TEST (100 phút)", value=st.session_state.is_test_mode, help="Bỏ chọn = Practice"
        )
        st.session_state.shuffle_q = st.checkbox("Xáo trộn câu", value=st.session_state.shuffle_q)
        st.session_state.shuffle_opt = st.checkbox("Xáo trộn đáp án", value=st.session_state.shuffle_opt)

        if st.button("Nạp câu hỏi", use_container_width=True):
            bank = [QAItem(**d) for d in _load_bank(code, st.session_state.module)]
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

        start_col1, start_col2 = st.columns(2)
        with start_col1:
            if st.button(
                "BẮT ĐẦU", type="primary", use_container_width=True, disabled=(len(st.session_state.bank) == 0)
            ):
                st.session_state.is_quiz_active = True
                st.session_state.cur = 0
                st.session_state.picks = {}
                st.session_state.fails_first_try = set()
                if st.session_state.is_test_mode:
                    st.session_state.start_time = time.time()
                    st.session_state.remaining = TEST_DURATION_SECONDS
                st.toast("Bắt đầu làm bài!")
        with start_col2:
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

    # Timer (only in test mode)
    if st.session_state.is_quiz_active and st.session_state.is_test_mode and st.session_state.start_time is not None:
        elapsed = int(time.time() - st.session_state.start_time)
        st.session_state.remaining = max(0, TEST_DURATION_SECONDS - elapsed)
        mm, ss = divmod(st.session_state.remaining, 60)
        st.markdown(f"### ⏱️ Thời gian còn lại: **{mm:02d}:{ss:02d}**")
    
        # Nếu hết giờ → nộp bài
        if st.session_state.remaining == 0:
            st.warning("⏰ Hết giờ! Hệ thống tự động nộp bài.")
            st.session_state.is_quiz_active = False
            st.session_state.is_test_mode = False
            st.session_state.start_time = None
            st.session_state.remaining = TEST_DURATION_SECONDS
            st.rerun()


    # ---------- Main Content ----------
    if st.session_state.is_quiz_active and st.session_state.bank:
        qi = st.session_state.order[st.session_state.cur]
        item: QAItem = st.session_state.bank[qi]

        st.caption(
            f"Câu {st.session_state.cur + 1}/{len(st.session_state.order)} • CAT {item.cat} • Module {item.module}"
        )
        st.write(f"**{item.q}**")

        options_indexed: List[Tuple[int, str]] = list(enumerate(item.options))
        if st.session_state.shuffle_opt:
            rnd = random.Random(qi)
            rnd.shuffle(options_indexed)

        picked = st.session_state.picks.get(qi, -1)
        correct_idx = item.answer

        def on_pick(choice: int):
            if not st.session_state.is_test_mode and qi not in st.session_state.picks and choice != correct_idx:
                st.session_state.fails_first_try.add(qi)
            st.session_state.picks[qi] = choice

        # Render options
        for original_idx, text in options_indexed:
            label = f"{chr(65+original_idx)}. {text}"
        
            # Kiểm tra nếu người dùng đã chọn
            if qi in st.session_state.picks and st.session_state.picks[qi] == original_idx:
                # Nếu là chế độ TEST: đánh dấu đáp án đã chọn
                label = f"✅ {label}"
                st.markdown(f"<div style='padding:8px;border-radius:6px;background-color:#e6f7ff;border:1px solid #91d5ff;'>{label}</div>", unsafe_allow_html=True)
            else:
                if st.button(label, key=f"opt_{qi}_{original_idx}", use_container_width=True):
                    on_pick(original_idx)
                    st.rerun()

        # Feedback area (Practice only)
        if not st.session_state.is_test_mode and picked != -1:
            if picked == correct_idx:
                st.success("ĐÚNG!")
            else:
                st.error("SAI! Vui lòng chọn lại.")

        # Translate (optional)
        if _HAS_TRANSLATOR and st.button("Dịch (VN)"):
            try:
                tr = Translator()
                q_tr = tr.translate(item.q, dest="vi").text
                opts_tr = []
                for i, opt in enumerate(item.options):
                    opts_tr.append(f"{chr(65+i)}. " + tr.translate(opt, dest="vi").text)
                st.info("**BẢN DỊCH CÂU HỎI:**\n" + q_tr + "\n\n---\n**BẢN DỊCH CÁC ĐÁP ÁN:**\n" + "\n".join(opts_tr))
            except Exception:
                st.warning("Lỗi dịch thuật (không có Internet hoặc bị giới hạn API)")
        elif not _HAS_TRANSLATOR:
            st.caption("Chức năng Dịch (VN) không khả dụng trên máy chủ này.")

        # Nav controls
        nav1, nav2, nav3, nav4 = st.columns(4)
        with nav1:
            if st.button("← Trước", disabled=(st.session_state.cur == 0)):
                st.session_state.cur -= 1
                st.rerun()
        with nav2:
            if st.button("Xóa chọn", disabled=(qi not in st.session_state.picks)):
                if qi in st.session_state.picks:
                    del st.session_state.picks[qi]
                if qi in st.session_state.fails_first_try:
                    st.session_state.fails_first_try.remove(qi)
                st.rerun()
        with nav3:
            can_next = (st.session_state.cur < len(st.session_state.order) - 1)
            if st.session_state.is_test_mode:
                next_disabled = not can_next
            else:
                # Practice: only allow next if correct
                next_disabled = not (can_next and picked == correct_idx)
            if st.button("Tiếp →", disabled=next_disabled):
                st.session_state.cur += 1
                st.rerun()
        with nav4:
            if st.button("Nộp bài", type="primary"):
                st.session_state.is_quiz_active = False
                st.rerun()

    # ---------- Results ----------
    if not st.session_state.is_quiz_active and st.session_state.bank and (st.session_state.picks or st.session_state.is_test_mode):
        right = 0
        rows = []
        for n, qi in enumerate(st.session_state.order, 1):
            it: QAItem = st.session_state.bank[qi]
            picked = st.session_state.picks.get(qi, None)
            ans_idx = it.answer
            is_right = (picked == ans_idx)
            right += int(bool(is_right))
            picked_text = it.options[picked] if picked is not None else "(bỏ trống)"
            correct_text = it.options[ans_idx]
            rows.append({
                "#": n,
                "CAT": it.cat,
                "Module": it.module,
                "Question": it.q,
                "Your Answer": picked_text,
                "Correct Answer": correct_text,
                "Status": "Đúng" if is_right else "Sai",
            })

        total = len(st.session_state.order)
        percent = round(100 * right / total) if total else 0
        st.subheader("Kết quả")
        st.write(f"**Đúng {right}/{total} ({percent}%)**")

        try:
            import pandas as pd  # optional
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Tải kết quả (CSV)", data=csv_bytes, file_name="results.csv", mime="text/csv")
        except Exception:
            st.write("Không thể hiển thị bảng hoặc tải CSV (thiếu pandas)")

        if st.button("Làm lại từ đầu"):
            st.session_state.bank = []
            st.session_state.order = []
            st.session_state.cur = 0
            st.session_state.picks = {}
            st.rerun()

    # ---------- Mixed Exam (MVP) ----------
    with st.expander("Tạo đề hỗn hợp (MVP)"):
        total_mix = st.number_input("Tổng số câu", min_value=10, max_value=300, value=100, step=10)
        colA, colB, colC = st.columns(3)
        with colA:
            pA = st.number_input("% B1", min_value=0, max_value=100, value=0)
        with colB:
            pB = st.number_input("% B2", min_value=0, max_value=100, value=0)
        with colC:
            pC = st.number_input("% M10", min_value=0, max_value=100, value=0)
        if st.button("Tạo đề hỗn hợp"):
            try:
                bank = mix_generate({"B1": pA, "B2": pB, "M10": pC}, int(total_mix))
                if not bank:
                    st.error("Không tạo được đề (dữ liệu rỗng)")
                else:
                    random.shuffle(bank)
                    st.session_state.bank = bank
                    st.session_state.order = list(range(len(bank)))
                    st.session_state.cur = 0
                    st.session_state.picks = {}
                    st.session_state.fails_first_try = set()
                    st.success(f"Đã tạo đề gồm {len(bank)} câu. Nhấn BẮT ĐẦU để làm bài.")
            except ValueError as e:
                st.error(str(e))


# -----------------------------------------------------------
# CLI Fallback (no external packages)
# -----------------------------------------------------------

def _input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def main_cli():
    print(f"\n{APP_TITLE} – CLI fallback (no streamlit)\n")
    cats = list(CATEGORIES.items())
    for i, (label, code) in enumerate(cats, 1):
        print(f"{i}. {label} [{code}]")
    while True:
        try:
            ci = int(_input("Chọn loại đề (1-3): ").strip() or "1")
            if 1 <= ci <= len(cats):
                break
        except Exception:
            pass
        print("Nhập không hợp lệ.")
    label, code = cats[ci - 1]

    mods = list_available_modules(code)
    if not mods:
        print("\n⚠️  Không tìm thấy CSV. Hãy đặt file {CAT}_Module{N}.csv cùng thư mục.")
        return
    print("\nModule có sẵn:", ", ".join(map(str, mods)))
    while True:
        m = _input(f"Chọn module ({mods[0]}-{mods[-1]}): ").strip() or str(mods[0])
        if m.isdigit() and int(m) in mods:
            module = m
            break
        print("Module không hợp lệ.")

    bank = load_csv_bank(code, module)
    if not bank:
        print("\n⚠️  Không có dữ liệu trong CSV.")
        return

    print(f"\nĐã nạp {len(bank)} câu. Nhập số câu muốn làm (<= {len(bank)}), ví dụ 10:")
    try:
        n = int(_input("Số câu: ").strip() or "10")
    except Exception:
        n = 10
    n = max(1, min(n, len(bank)))

    rng = random.Random()
    rng.shuffle(bank)
    picked = {}
    correct = 0
    for idx, item in enumerate(bank[:n], 1):
        print(f"\nCâu {idx}/{n} – CAT {item.cat} – Module {item.module}")
        print(item.q)
        for i, opt in enumerate(item.options):
            print(f"  {chr(65+i)}. {opt}")
        ans = _input("Chọn (A/B/C/D... hoặc Enter để bỏ): ").strip().upper()
        if ans:
            choice = ord(ans[0]) - 65
            picked[idx] = choice
            if choice == item.answer:
                print("→ ĐÚNG!")
                correct += 1
            else:
                print(f"→ SAI! Đáp án đúng: {chr(65+item.answer)}")
        else:
            print(f"→ BỎ TRỐNG! Đáp án đúng: {chr(65+item.answer)}")

    percent = round(100 * correct / n)
    print(f"\nKẾT QUẢ: {correct}/{n} đúng ({percent}%).\n")


# -----------------------------------------------------------
# Self Tests (no external packages)
# -----------------------------------------------------------

def run_self_tests() -> None:
    print("Running self tests...")
    tmp = tempfile.TemporaryDirectory()
    base = tmp.name

    # Create minimal banks for B1 (Module1) and M10 (Module2)
    def make_csv(cat: str, module: int, rows: List[Tuple[str, List[str], str]]):
        path = os.path.join(base, f"{cat}_Module{module}.csv")
        headers = ["Question", "Option A", "Option B", "Option C", "Correct Answer"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for q, opts, correct in rows:
                row = [q]
                # ensure at least 3 columns for options
                for i in range(3):
                    row.append(opts[i] if i < len(opts) else "")
                row.append(correct)
                w.writerow(row)

    make_csv("B1", 1, [
        ("Q1 B1?", ["A1", "B1", "C1"], "B1"),
        ("Q2 B1?", ["A2", "B2", "C2"], "A2"),
    ])
    make_csv("M10", 2, [
        ("Q1 M10?", ["X1", "Y1", "Z1"], "Z1"),
        ("Q2 M10?", ["X2", "Y2", "Z2"], "Y2"),
        ("Q3 M10?", ["X3", "Y3", "Z3"], "X3"),
    ])

    # Test 1: list_available_modules
    assert list_available_modules("B1", base) == [1], "Modules for B1 should be [1]"
    assert list_available_modules("M10", base) == [2], "Modules for M10 should be [2]"

    # Test 2: load_csv_bank structure
    b1 = load_csv_bank("B1", "1", base)
    assert len(b1) == 2 and b1[0].q.startswith("Q"), "load_csv_bank failed for B1"
    assert any("B1" in opt for opt in b1[0].options), "Options not parsed"

    # Test 3: mix_generate 50/50 of total 5 → expect 2 from B1 and 3 from M10 (largest remainder)
    mixed = mix_generate({"B1": 50, "M10": 50}, 5, base)
    assert len(mixed) == 5, "Mixed length must equal total"
    cats = [x.cat for x in mixed]
    assert cats.count("B1") == 2 and cats.count("M10") == 3, "Category distribution mismatch (expected 2/3)"

    # Test 4: cap by availability – ask 10 with 50/50 but we only have 2 and 3 → expect 5 total
    mixed2 = mix_generate({"B1": 50, "M10": 50}, 10, base)
    assert len(mixed2) == 5, "Should cap to available when requesting more than availability"

    print("Self tests passed.\n")


# -----------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--cli", action="store_true", help="Run CLI fallback (no dependencies)")
    parser.add_argument("--selftest", action="store_true", help="Run built-in tests (no dependencies)")
    args = parser.parse_args()

    if args.selftest:
        run_self_tests()
        sys.exit(0)

    # Default behavior:
    # - If streamlit is available → run Streamlit UI (no need for QUIZ_MODE)
    # - If user explicitly passes --cli or streamlit is not available → run CLI
    if args.cli or _st is None:
        if _st is None:
            print("[INFO] streamlit not found → running CLI fallback. Install streamlit to use the web UI.")
        main_cli()
    else:
        main_streamlit()
