def main_streamlit():
    st = _st
    if st is None:
        raise RuntimeError("Streamlit not available; run CLI mode or install streamlit.")

    st.title("Quiz CAAV Cat B (Web)")

    # ---------------- Sidebar (luôn hiển thị) ----------------
    with st.sidebar:
        st.header("Cấu hình")
        ss = st.session_state
        ss.category = ss.get("category", list(CATEGORIES.keys())[0])
        ss.module = ss.get("module", MODULES[0])
        ss.is_test_mode = ss.get("is_test_mode", False)
        ss.shuffle_q = ss.get("shuffle_q", False)
        ss.shuffle_opt = ss.get("shuffle_opt", False)
        ss.bank = ss.get("bank", [])
        ss.order = ss.get("order", [])
        ss.cur = ss.get("cur", 0)
        ss.picks = ss.get("picks", {})
        ss.fails_first_try = ss.get("fails_first_try", set())
        ss.start_time = ss.get("start_time", None)
        ss.remaining = ss.get("remaining", TEST_DURATION_SECONDS)

        ss.category = st.selectbox("Loại đề", list(CATEGORIES.keys()),
                                   index=list(CATEGORIES.keys()).index(ss.category))
        code = CATEGORIES[ss.category]

        avail = list_available_modules(code)
        if avail:
            ss.module = st.selectbox("Module", [str(x) for x in avail], index=0)
        else:
            ss.module = st.selectbox("Module (không tìm thấy file)", MODULES, index=0)

        ss.is_test_mode = st.toggle("Chế độ TEST (100 phút)", value=ss.is_test_mode, help="Bỏ chọn = Practice")
        ss.shuffle_q = st.checkbox("Xáo trộn câu", value=ss.shuffle_q)
        ss.shuffle_opt = st.checkbox("Xáo trộn đáp án", value=ss.shuffle_opt)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Nạp câu hỏi", use_container_width=True):
                bank = load_csv_bank(code, ss.module)
                ss.bank = bank
                if not bank:
                    st.warning("Không tìm thấy dữ liệu.")
                else:
                    ss.order = list(range(len(bank)))
                    if ss.shuffle_q:
                        random.shuffle(ss.order)
                    ss.cur = 0
                    ss.picks = {}
                    ss.fails_first_try = set()
                    st.success(f"Đã nạp {len(bank)} câu từ {code}_Module{ss.module}.csv")
        with col2:
            if st.button("BẮT ĐẦU", type="primary", use_container_width=True, disabled=(len(ss.bank) == 0)):
                ss.is_quiz_active = True
                ss.cur = 0
                ss.picks = {}
                ss.fails_first_try = set()
                if ss.is_test_mode:
                    ss.start_time = time.time()
                    ss.remaining = TEST_DURATION_SECONDS
                st.toast("Bắt đầu làm bài!")

        if st.button("HỦY THI", use_container_width=True, disabled=not ss.get("is_quiz_active", False)):
            ss.is_quiz_active = False
            ss.bank = []
            ss.order = []
            ss.picks = {}
            ss.fails_first_try = set()
            ss.cur = 0
            ss.start_time = None
            ss.remaining = TEST_DURATION_SECONDS
            st.info("Đã hủy bài thi.")

        st.markdown("---")
        st.subheader("🏗️ Tạo đề hỗn hợp")
        total_mix = st.number_input("Tổng số câu", min_value=10, max_value=300, value=100, step=10, key="mix_total")
        c1, c2, c3 = st.columns(3)
        with c1: pA = st.number_input("% B1", 0, 100, 0, key="mix_p_b1")
        with c2: pB = st.number_input("% B2", 0, 100, 0, key="mix_p_b2")
        with c3: pC = st.number_input("% M10", 0, 100, 0, key="mix_p_m10")
        if st.button("Tính phân bổ", use_container_width=True):
            st.session_state._mix_plan = True
        if st.button("Tạo đề hỗn hợp", use_container_width=True):
            if pA + pB + pC != 100:
                st.error("Tổng % phải = 100")
            else:
                def _plan_distribution():
                    plan = {}
                    for cat, pct in [("B1", pA), ("B2", pB), ("M10", pC)]:
                        if pct <= 0: continue
                        cat_need = round(int(total_mix) * pct / 100)
                        mods = list_available_modules(cat)
                        rows = []
                        if not mods:
                            plan[cat] = [{"module": "-", "available": 0, "need": 0, "take": 0}]
                            continue
                        per_mod = cat_need // len(mods); rem = cat_need % len(mods)
                        for i, m in enumerate(mods):
                            need = per_mod + (1 if i < rem else 0)
                            avail = len(load_csv_bank(cat, str(m)))
                            rows.append({"module": m, "available": avail, "need": need, "take": min(need, avail)})
                        plan[cat] = rows
                    return plan

                plan = _plan_distribution()
                rng = random.Random()
                bank = []
                for cat, rows in plan.items():
                    for r in rows:
                        if r["take"] <= 0 or r["module"] == "-": continue
                        pool = load_csv_bank(cat, str(r["module"]))
                        bank.extend(rng.sample(pool, min(r["take"], len(pool))))
                if not bank:
                    st.error("Không tạo được đề (dữ liệu rỗng)")
                else:
                    rng.shuffle(bank)
                    ss.bank = bank; ss.order = list(range(len(bank))); ss.cur = 0
                    ss.picks = {}; ss.fails_first_try = set()
                    st.success("Đã tạo đề. Vào tab **🧩 Làm đề** và bấm **BẮT ĐẦU** để thi.")

    # ---------------- Tabs (nội dung chính & báo cáo) ----------------
    tab_main, tab_mix = st.tabs(["🧩 Làm đề", "🏗️ Báo cáo phân bổ"])

    # ===== Tab: Làm đề =====
    with tab_main:
        ss = st.session_state
        # Timer (không rerun liên tục)
        if ss.get("is_quiz_active") and ss.is_test_mode and ss.start_time is not None:
            elapsed = int(time.time() - ss.start_time)
            ss.remaining = max(0, TEST_DURATION_SECONDS - elapsed)
            mm, ss2 = divmod(ss.remaining, 60)
            st.markdown(f"### ⏱️ Thời gian còn lại: **{mm:02d}:{ss2:02d}**")
            if ss.remaining == 0:
                st.warning("⏰ Hết giờ! Hệ thống tự động nộp bài.")
                ss.is_quiz_active = False
                ss.is_test_mode = False
                ss.start_time = None
                ss.remaining = TEST_DURATION_SECONDS
                st.rerun()

        # Hiển thị câu hỏi
        if ss.get("is_quiz_active") and ss.bank:
            qi = st.session_state.order[st.session_state.cur]
            item = st.session_state.bank[qi]
            st.caption(f"Câu {st.session_state.cur + 1}/{len(st.session_state.order)} • CAT {item.cat} • Module {item.module}")
            st.write(f"**{item.q}**")

            options_indexed = list(enumerate(item.options))
            if st.session_state.shuffle_opt:
                rnd = random.Random(qi); rnd.shuffle(options_indexed)

            picked = st.session_state.picks.get(qi, -1)
            correct_idx = item.answer

            def on_pick(choice: int):
                if not st.session_state.is_test_mode and qi not in st.session_state.picks and choice != correct_idx:
                    st.session_state.fails_first_try.add(qi)
                st.session_state.picks[qi] = choice

            for original_idx, text in options_indexed:
                label = f"{chr(65+original_idx)}. {text}"
                if qi in st.session_state.picks and st.session_state.picks[qi] == original_idx and st.session_state.is_test_mode:
                    st.markdown(
                        f"<div style='padding:8px;border-radius:6px;background:#e6f7ff;border:1px solid #91d5ff'>✅ {label}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(label, key=f"opt_{qi}_{original_idx}", use_container_width=True):
                        on_pick(original_idx); st.rerun()

            if not st.session_state.is_test_mode and picked != -1:
                st.success("ĐÚNG!") if picked == correct_idx else st.error("SAI! Vui lòng chọn lại.")

            nav1, nav2, nav3, nav4 = st.columns(4)
            with nav1:
                if st.button("← Trước", disabled=(st.session_state.cur == 0)):
                    st.session_state.cur -= 1; st.rerun()
            with nav2:
                if st.button("Xóa chọn", disabled=(qi not in st.session_state.picks)):
                    st.session_state.picks.pop(qi, None); st.session_state.fails_first_try.discard(qi); st.rerun()
            with nav3:
                can_next = (st.session_state.cur < len(st.session_state.order) - 1)
                next_disabled = not can_next if st.session_state.is_test_mode else not (can_next and picked == correct_idx)
                if st.button("Tiếp →", disabled=next_disabled):
                    st.session_state.cur += 1; st.rerun()
            with nav4:
                if st.button("Nộp bài", type="primary"):
                    st.session_state.is_quiz_active = False; st.rerun()

        # Kết quả
        if (not st.session_state.get("is_quiz_active")) and st.session_state.bank and (st.session_state.picks or st.session_state.is_test_mode):
            right = 0; rows = []
            for n, qi in enumerate(st.session_state.order, 1):
                it = st.session_state.bank[qi]
                picked = st.session_state.picks.get(qi, None)
                ans_idx = it.answer; is_right = (picked == ans_idx); right += int(bool(is_right))
                rows.append({
                    "#": n, "CAT": it.cat, "Module": it.module,
                    "Question": it.q,
                    "Your Answer": it.options[picked] if picked is not None else "(bỏ trống)",
                    "Correct Answer": it.options[ans_idx],
                    "Status": "Đúng" if is_right else "Sai",
                })
            total = len(st.session_state.order); percent = round(100 * right / total) if total else 0
            st.subheader("Kết quả"); st.write(f"**Đúng {right}/{total} ({percent}%)**")
            try:
                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                st.download_button("Tải kết quả (CSV)", data=df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="results.csv", mime="text/csv")
            except Exception:
                pass

    # ===== Tab: Báo cáo phân bổ (nếu vừa bấm 'Tính phân bổ') =====
    with tab_mix:
        if getattr(st.session_state, "_mix_plan", False):
            import pandas as pd
            st.subheader("Báo cáo phân bổ theo module")
            pA = st.session_state.get("mix_p_b1", 0); pB = st.session_state.get("mix_p_b2", 0); pC = st.session_state.get("mix_p_m10", 0)
            total_mix = st.session_state.get("mix_total", 100)
            total_take = 0
            for cat, pct in [("B1", pA), ("B2", pB), ("M10", pC)]:
                if pct <= 0: continue
                cat_need = round(int(total_mix) * pct / 100)
                mods = list_available_modules(cat)
                rows = []
                per_mod = cat_need // len(mods) if mods else 0
                rem = cat_need % len(mods) if mods else 0
                for i, m in enumerate(mods or []):
                    need = per_mod + (1 if i < rem else 0)
                    avail = len(load_csv_bank(cat, str(m)))
                    rows.append({"Module": m, "Có sẵn": avail, "Cần": need, "Lấy": min(need, avail)})
                if rows:
                    df = pd.DataFrame(rows); st.markdown(f"**[{cat}] Tổng dự kiến:** {df['Lấy'].sum()} câu")
                    st.dataframe(df, use_container_width=True); total_take += int(df["Lấy"].sum())
            st.info(f"Tổng cộng sẽ lấy: **{total_take}**/{int(total_mix)} câu.")
# --- Run Streamlit UI by default ---
if _st is not None:
    main_streamlit()
