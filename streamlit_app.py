import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import date, datetime, timedelta
import uuid

# ==============================================================================
# 1. CONFIG & SYSTEM SETUP
# ==============================================================================
st.set_page_config(page_title="Planejador de Obra", layout="wide", page_icon="🏗️")

# File Paths
FILES = {
    "tx": Path("transactions.csv"),
    "obj": Path("objectives.csv"),
    "alloc": Path("allocations.csv"),
    "rec": Path("recurring.csv"),
    "conf": Path("config.csv")
}

# Schemas (Strict Typing)
SCHEMAS = {
    "tx": {
        "id": "str", "date": "datetime64[ns]", "type": "str", "category": "str", 
        "amount": "float64", "notes": "str", "recurring_id": "str"
    },
    "obj": {
        "id": "str", "name": "str", "target": "float64", "priority": "int64", 
        "phase": "str", "floor": "str", "due_date": "datetime64[ns]", 
        "status": "str", "notes": "str"
    },
    "alloc": {
        "tx_id": "str", "objective_id": "str", "amount": "float64"
    },
    "rec": {
        "id": "str", "name": "str", "type": "str", "category": "str", 
        "amount": "float64", "frequency": "str", 
        "start_date": "datetime64[ns]", "next_date": "datetime64[ns]", 
        "active": "bool"
    },
    "conf": {
        "key": "str", "value": "str"
    }
}

FLOOR_OPTIONS = ["Térreo", "1º Andar", "Cobertura", "Área Externa", "Geral"]

# ==============================================================================
# 2. HELPER FUNCTIONS (ROBUSTNESS)
# ==============================================================================

def safe_format_date(dt_val):
    """Prevents NaTType errors by checking validity first."""
    if pd.isnull(dt_val) or dt_val is pd.NaT:
        return "??"
    try:
        return dt_val.strftime('%d/%m')
    except:
        return "??"

def format_currency(value):
    """Converts 1234.56 to 'R$ 1.234,56' safely."""
    if pd.isna(value): return "R$ 0,00"
    try:
        return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "R$ 0,00"

def clean_id_column(series):
    """Removes floating point decimals from string IDs (e.g. '1.0' -> '1')."""
    return series.astype(str).str.replace(r'\.0$', '', regex=True)

def load_dataset_safe(key):
    """Loads CSV with strict error handling and schema enforcement."""
    path = FILES[key]
    schema = SCHEMAS[key]
    
    try:
        if path.exists():
            # Load CSV, but don't parse dates yet to avoid crashes on bad formats
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=schema.keys())
    except Exception as e:
        st.error(f"Erro ao ler {key}: {e}")
        df = pd.DataFrame(columns=schema.keys())

    # 1. Ensure all expected columns exist
    for col, dtype in schema.items():
        if col not in df.columns:
            if "float" in dtype or "int" in dtype: default = 0
            elif "bool" in dtype: default = True
            elif col == "floor": default = "Térreo"
            else: default = ""
            df[col] = default

    # 2. Enforce Data Types
    for col, dtype in schema.items():
        try:
            if "datetime" in dtype:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif "float" in dtype:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif "int" in dtype:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            elif "bool" in dtype:
                df[col] = df[col].astype(bool)
            else:
                df[col] = df[col].astype(str)
        except:
            pass # Keep going if a single column fails

    # 3. Clean IDs
    for col in ["id", "objective_id", "tx_id", "recurring_id"]:
        if col in df.columns:
            df[col] = clean_id_column(df[col])

    # Reorder columns to match schema
    return df[list(schema.keys())]

def save_data(key, df):
    """Saves DataFrame to CSV."""
    try:
        df.to_csv(FILES[key], index=False)
    except Exception as e:
        st.error(f"Erro ao salvar {key}: {e}")

def generate_id(df):
    """Generates a robust unique ID."""
    if df.empty: return "1"
    
    # Try numeric increment first
    try:
        ids = [int(i) for i in df["id"] if str(i).isdigit()]
        if ids:
            return str(max(ids) + 1)
    except:
        pass
        
    # Fallback to UUID if numeric fails
    return str(uuid.uuid4())[:8]

def get_setting(df, key, default):
    if df.empty: return default
    row = df[df["key"] == key]
    if not row.empty:
        return row.iloc[0]["value"]
    return default

def set_setting(df, key, value):
    df = df[df["key"] != key]
    new_row = {"key": key, "value": str(value)}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data("conf", df)
    return df

# Load Data
tx_df, obj_df, alloc_df, rec_df, conf_df = (
    load_dataset_safe("tx"),
    load_dataset_safe("obj"),
    load_dataset_safe("alloc"),
    load_dataset_safe("rec"),
    load_dataset_safe("conf")
)

# ==============================================================================
# 3. SIDEBAR
# ==============================================================================
st.sidebar.title("🏗️ Planejador")
page = st.sidebar.radio(
    "Navegação", 
    ["⚫ Visão Geral", "💳 Fluxo de Caixa", "🏁 Objetivos", "🧱 Alocações"]
)

# Initialize Session State
if "edit_obj_id" not in st.session_state: st.session_state.edit_obj_id = None
if "edit_rec_id" not in st.session_state: st.session_state.edit_rec_id = None

# ==============================================================================
# 4. PAGE: OVERVIEW
# ==============================================================================
if page == "⚫ Visão Geral":
    st.title("⚫ Painel de Controle")

    st.subheader(f"📅 Desempenho Mensal ({date.today().strftime('%m/%Y')})")
    
    # Logic
    monthly_goal = float(get_setting(conf_df, "monthly_savings_goal", "5000"))
    current_month = date.today().month
    current_year = date.today().year
    
    tx_month_ids = tx_df[
        (tx_df["date"].dt.month == current_month) & 
        (tx_df["date"].dt.year == current_year)
    ]["id"].unique()
    
    month_saved = 0.0
    if not alloc_df.empty:
        month_saved = alloc_df[alloc_df["tx_id"].isin(tx_month_ids)]["amount"].sum()
    
    remaining_month = monthly_goal - month_saved
    month_progress = min(month_saved / monthly_goal, 1.0) if monthly_goal > 0 else 0

    # UI
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
    
    with c1:
        st.metric("Meta Mensal", format_currency(monthly_goal))
        with st.popover("⚙️ Ajustar Meta"):
            new_goal = st.number_input("Definir nova meta mensal", value=monthly_goal, step=500.0)
            if st.button("💾 Salvar"):
                conf_df = set_setting(conf_df, "monthly_savings_goal", new_goal)
                st.rerun()
                
    with c2:
        st.metric("Economizado (Mês)", format_currency(month_saved), delta=f"{month_progress*100:.0f}% da meta")
        
    with c3:
        st.metric("Falta para Meta", format_currency(max(remaining_month, 0)), delta_color="inverse")
        
    with c4:
        st.write("**Progresso da Meta Mensal**")
        st.progress(month_progress)
        if month_saved >= monthly_goal:
            st.caption("✔️ Meta mensal atingida!")

    st.markdown("---")

    # Global Stats
    st.subheader("🏗️ Status Global da Obra")
    
    if not alloc_df.empty:
        alloc_summary = alloc_df.groupby("objective_id")["amount"].sum().reset_index()
        alloc_summary.columns = ["id", "allocated"]
        alloc_summary["id"] = clean_id_column(alloc_summary["id"])
        merged = obj_df.merge(alloc_summary, on="id", how="left")
        merged["allocated"] = merged["allocated"].fillna(0)
    else:
        merged = obj_df.copy()
        merged["allocated"] = 0.0

    total_target = merged["target"].sum()
    total_alloc = merged["allocated"].sum()
    total_remaining = total_target - total_alloc
    global_fin_pct = (total_alloc / total_target) if total_target > 0 else 0

    total_objs = len(merged)
    completed_objs = len(merged[merged["status"] == "concluído"])
    global_obj_pct = (completed_objs / total_objs) if total_objs > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orçamento Total", format_currency(total_target))
    k2.metric("Total Pago/Guardado", format_currency(total_alloc))
    k3.metric("Falta Alocar", format_currency(total_remaining), delta_color="inverse")
    k4.metric("Conclusão Financeira", f"{global_fin_pct*100:.1f}%")

    st.write("**Progresso das Etapas (Objetivos Concluídos)**")
    st.progress(global_obj_pct)
    st.caption(f"{completed_objs} de {total_objs} etapas concluídas ({global_obj_pct*100:.1f}%)")

    st.markdown("---")
    col_graph, col_prio = st.columns([1, 1])

    with col_graph:
        st.markdown("### 📊 Análise Visual")
        st.write("**Progresso Financeiro**")
        donut_data = pd.DataFrame([
            {"category": "Guardado", "value": total_alloc},
            {"category": "Falta", "value": total_remaining}
        ])
        
        base = alt.Chart(donut_data).encode(theta=alt.Theta("value", stack=True))
        pie = base.mark_arc(outerRadius=100, innerRadius=60).encode(
            color=alt.Color("category", scale=alt.Scale(domain=["Guardado", "Falta"], range=["#2E2E2E", "#B0B0B0"])),
            tooltip=["category", alt.Tooltip("value", format=",.2f")]
        )
        st.altair_chart(pie, use_container_width=True)

    with col_prio:
        st.markdown("### 🏆 Ranking de Prioridades")
        if not merged.empty:
            active_objs = merged[merged["status"] != "concluído"].sort_values("priority", ascending=True)
            if active_objs.empty:
                st.success("🎉 Nenhum objetivo pendente!")
            else:
                for i, row in active_objs.head(5).iterrows():
                    pct = (row["allocated"] / row["target"]) if row["target"] > 0 else 0
                    with st.container():
                        st.markdown(f"**{row['priority']}. {row['name']}**")
                        st.progress(min(pct, 1.0))
                        st.caption(f"Meta: {format_currency(row['target'])} • Restam: {format_currency(row['target'] - row['allocated'])}")
        else:
            st.info("Cadastre objetivos para ver o ranking.")

# ==============================================================================
# 5. PAGE: CASH FLOW
# ==============================================================================
elif page == "💳 Fluxo de Caixa":
    st.title("💳 Fluxo de Caixa")

    # Metrics
    income = tx_df[tx_df["type"] == "income"]["amount"].sum()
    expense = tx_df[tx_df["type"] == "expense"]["amount"].sum()
    net = income - expense
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Entradas Totais", format_currency(income), delta="Total")
    m2.metric("Saídas Totais", format_currency(expense), delta="-Gastos", delta_color="inverse")
    m3.metric("Saldo Líquido", format_currency(net))
    
    st.markdown("---")

    # Split Layout
    col_data, col_action = st.columns([1.8, 1.2], gap="large")

    # --- LEFT: STATEMENT ---
    with col_data:
        st.subheader("📄 Extrato")
        
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1: f_search = st.text_input("🔎 Buscar (Nome/Categoria)", placeholder="Ex: Cimento...")
        with fc2: f_month = st.date_input("Filtrar Data", value=[])
        with fc3: f_type_filter = st.selectbox("Tipo", ["Todos", "Entrada", "Saída"])
        
        display_df = tx_df.copy()
        if f_search:
            # Safe string handling
            mask = display_df["category"].astype(str).str.contains(f_search, case=False, na=False) | \
                   display_df["notes"].astype(str).str.contains(f_search, case=False, na=False)
            display_df = display_df[mask]
            
        if f_type_filter != "Todos":
            display_df = display_df[display_df["type"] == ("income" if f_type_filter == "Entrada" else "expense")]

        display_df = display_df.sort_values("date", ascending=False)
        
        # Display Table
        formatted_df = display_df.copy()
        formatted_df["amount_display"] = formatted_df["amount"].apply(format_currency)
        formatted_df["date_display"] = formatted_df["date"].dt.strftime('%d/%m/%Y')
        
        st.dataframe(
            formatted_df[["date", "type", "category", "amount", "notes"]],
            column_config={
                "date": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                "type": "Tipo", "category": "Categoria", "amount": "Valor", "notes": "Notas"
            },
            use_container_width=True, hide_index=True, height=400
        )
        
        # Edit Transaction
        with st.expander("🖊️ Editar ou Excluir Lançamento Selecionado"):
             if tx_df.empty:
                st.info("Nada para editar.")
             else:
                tx_df_sorted = tx_df.sort_values("date", ascending=False)
                # Create Unique Label Map: {Label: ID} to prevent duplicates
                tx_options = {
                    f"{row['date'].strftime('%d/%m')} | {format_currency(row['amount'])} | {row['category']} (ID:{row['id']})": row['id'] 
                    for _, row in tx_df_sorted.iterrows()
                }
                
                sel_label = st.selectbox("Selecione o item:", list(tx_options.keys()))
                
                if sel_label:
                    sel_id = tx_options[sel_label]
                    tx_to_edit = tx_df[tx_df["id"] == sel_id].iloc[0]
                    
                    with st.form("edit_tx_form_sleek"):
                        ec1, ec2 = st.columns(2)
                        et_date = ec1.date_input("Data", tx_to_edit["date"])
                        et_amount = ec2.number_input("Valor", value=float(tx_to_edit["amount"]), min_value=0.0)
                        et_cat = st.text_input("Categoria", tx_to_edit["category"])
                        et_notes = st.text_input("Descrição", tx_to_edit["notes"])
                        
                        esc1, esc2 = st.columns([1, 4])
                        if esc1.form_submit_button("💾 Salvar"):
                            tx_df.loc[tx_df["id"] == sel_id, ["date", "category", "amount", "notes"]] = [et_date, et_cat, et_amount, et_notes]
                            save_data("tx", tx_df)
                            st.success("Atualizado!")
                            st.rerun()
                        if st.form_submit_button("✖️ Remover", type="primary"):
                            # Cascade delete allocations to avoid orphan data
                            alloc_df = alloc_df[alloc_df["tx_id"] != str(sel_id)]
                            tx_df = tx_df[tx_df["id"] != sel_id]
                            save_data("alloc", alloc_df)
                            save_data("tx", tx_df)
                            st.warning("Removido.")
                            st.rerun()

    # --- RIGHT: ACTIONS ---
    with col_action:
        # 1. New Transaction
        with st.container(border=True):
            st.markdown("### 💸 Novo Lançamento")
            with st.form("quick_add_tx"):
                qa_type = st.segmented_control("Tipo", ["income", "expense"], selection_mode="single", default="expense", format_func=lambda x: "Entrada" if x=="income" else "Saída")
                
                qa_d1, qa_d2 = st.columns(2)
                qa_amount = qa_d1.number_input("Valor", min_value=0.0, step=50.0)
                qa_date = qa_d2.date_input("Data", date.today())
                qa_cat = st.text_input("Categoria", "Geral")
                qa_desc = st.text_input("Descrição (Opcional)")
                
                if st.form_submit_button("✔️ Confirmar", use_container_width=True):
                    new_tx = {
                        "id": generate_id(tx_df), "date": qa_date, 
                        "type": qa_type if qa_type else "expense", 
                        "category": qa_cat, "amount": qa_amount, 
                        "notes": qa_desc, "recurring_id": None
                    }
                    tx_df = pd.concat([tx_df, pd.DataFrame([new_tx])], ignore_index=True)
                    save_data("tx", tx_df)
                    st.toast("Lançamento salvo com sucesso!")
                    st.rerun()

        st.markdown("") 

        # 2. Recurring
        with st.container(border=True):
            st.markdown("### 🔄 Recorrências (A Vencer)")
            
            if not rec_df.empty:
                # Filter active
                active_recs = rec_df[rec_df["active"] == True].copy()
                
                # SAFETY: Handle missing next_dates by filling with today
                active_recs["next_date"] = pd.to_datetime(active_recs["next_date"], errors="coerce")
                active_recs["next_date"] = active_recs["next_date"].fillna(pd.to_datetime(date.today()))
                
                active_recs = active_recs.sort_values("next_date")
                
                if active_recs.empty:
                     st.info("Nenhuma conta recorrente ativa.")
                else:
                    for _, r in active_recs.iterrows():
                        rec_col1, rec_col2 = st.columns([3, 1])
                        with rec_col1:
                            # SAFETY: Use helper function
                            d_str = safe_format_date(r['next_date'])
                            type_icon = "⬇️" if r['type'] == 'expense' else "⬆️"
                            st.caption(f"{d_str} • {r['category']} {type_icon}")
                            st.write(f"**{r['name']}**")
                        with rec_col2:
                            st.write(f"**{format_currency(r['amount'])}**")
                            if st.button("✔️", key=f"btn_rec_{r['id']}", help="Lançar agora"):
                                new_tx = {
                                    "id": generate_id(tx_df), "date": r["next_date"], "type": r["type"],
                                    "category": r["category"], "amount": r["amount"], 
                                    "notes": f"Auto: {r['name']}", "recurring_id": r["id"]
                                }
                                tx_df = pd.concat([tx_df, pd.DataFrame([new_tx])], ignore_index=True)
                                
                                # Calc next date safely
                                freq_map = {"weekly": 7, "biweekly": 14, "monthly": 30}
                                days_add = freq_map.get(r["frequency"], 30) # Default to 30 if error
                                rec_df.loc[rec_df["id"] == r["id"], "next_date"] = r["next_date"] + timedelta(days=days_add)
                                
                                save_data("tx", tx_df)
                                save_data("rec", rec_df)
                                st.toast("Lançado!")
                                st.rerun()
                        st.divider()
            else:
                st.info("Nenhuma conta recorrente.")

            with st.expander("🔧 Gerenciar Recorrências"):
                 st.caption("Nova Recorrência")
                 with st.form("mini_rec_add"):
                     mr_name = st.text_input("Nome")
                     c_mr1, c_mr2 = st.columns(2)
                     mr_val = c_mr1.number_input("Valor", min_value=0.0)
                     mr_type = c_mr2.selectbox("Tipo", ["expense", "income"], format_func=lambda x: "Saída" if x=="expense" else "Entrada")
                     
                     mr_freq = st.selectbox("Freq.", ["monthly", "weekly"])
                     mr_start = st.date_input("Início", date.today())
                     
                     if st.form_submit_button("Add"):
                         new_rec = {
                            "id": generate_id(rec_df), "name": mr_name, 
                            "type": mr_type,
                            "category": "Fixo", "amount": mr_val, "frequency": mr_freq, 
                            "start_date": str(mr_start), "next_date": str(mr_start), "active": True
                         }
                         rec_df = pd.concat([rec_df, pd.DataFrame([new_rec])], ignore_index=True)
                         save_data("rec", rec_df)
                         st.rerun()
                 
                 st.caption("Remover Existente")
                 if not rec_df.empty:
                     rec_options = {f"{r['name']} ({format_currency(r['amount'])})": r['name'] for _, r in rec_df.iterrows()}
                     rec_sel = st.selectbox("Selecionar", list(rec_options.keys()))
                     if st.button("Remover Selecionado"):
                         name_to_del = rec_options[rec_sel]
                         rec_df = rec_df[rec_df["name"] != name_to_del]
                         save_data("rec", rec_df)
                         st.rerun()

# ==============================================================================
# 6. PAGE: OBJECTIVES
# ==============================================================================
elif page == "🏁 Objetivos":
    st.title("🏁 Objetivos da Obra")

    # Handle Edit State
    if st.session_state.edit_obj_id:
        with st.container():
            st.info(f"🖊️ Editando Objetivo ID: {st.session_state.edit_obj_id}")
            # Filter safely
            obj_filtered = obj_df[obj_df["id"] == st.session_state.edit_obj_id]
            if not obj_filtered.empty:
                curr_row = obj_filtered.iloc[0]
                
                with st.form("edit_obj_form"):
                    e1, e2, e3 = st.columns(3)
                    en_name = e1.text_input("Nome", curr_row["name"])
                    curr_fl = curr_row["floor"] if curr_row["floor"] in FLOOR_OPTIONS else "Térreo"
                    en_floor = e2.selectbox("Andar", FLOOR_OPTIONS, index=FLOOR_OPTIONS.index(curr_fl))
                    en_target = e3.number_input("Meta (R$)", value=float(curr_row["target"]))
                    
                    e4, e5 = st.columns(2)
                    en_prio = e4.number_input("Prioridade (1=Alta)", value=int(curr_row["priority"]), min_value=1, max_value=10)
                    en_status = e5.selectbox("Status", ["ativo", "concluído", "pausado"], index=["ativo", "concluído", "pausado"].index(curr_row["status"]))
                    
                    c_save, c_del, c_can = st.columns([1, 1, 4])
                    if c_save.form_submit_button("💾 Salvar"):
                        obj_df.loc[obj_df["id"] == st.session_state.edit_obj_id, ["name", "floor", "target", "priority", "status"]] = [en_name, en_floor, en_target, en_prio, en_status]
                        save_data("obj", obj_df)
                        st.session_state.edit_obj_id = None
                        st.rerun()
                    if c_del.form_submit_button("✖️ Excluir", type="primary"):
                        alloc_df = alloc_df[alloc_df["objective_id"] != st.session_state.edit_obj_id]
                        obj_df = obj_df[obj_df["id"] != st.session_state.edit_obj_id]
                        save_data("alloc", alloc_df)
                        save_data("obj", obj_df)
                        st.session_state.edit_obj_id = None
                        st.rerun()

            if st.button("✖️ Cancelar"):
                st.session_state.edit_obj_id = None
                st.rerun()
            st.divider()

    with st.expander("➕ Novo Objetivo", expanded=False):
        with st.form("new_obj"):
            c1, c2 = st.columns(2)
            n_name = c1.text_input("Nome (ex: Pisos)")
            n_floor = c2.selectbox("Andar", FLOOR_OPTIONS)
            c3, c4 = st.columns(2)
            n_target = c3.number_input("Meta (R$)", step=1000.0)
            n_prio = c4.number_input("Prioridade (1=Mais Alta)", min_value=1, max_value=10, value=5)
            
            if st.form_submit_button("✔️ Criar Objetivo"):
                if n_name and n_target > 0:
                    new_obj = {
                        "id": generate_id(obj_df), "name": n_name, "target": n_target, 
                        "priority": n_prio, "phase": "Geral", "floor": n_floor, 
                        "due_date": "", "status": "ativo", "notes": ""
                    }
                    obj_df = pd.concat([obj_df, pd.DataFrame([new_obj])], ignore_index=True)
                    save_data("obj", obj_df)
                    st.success("Criado!")
                    st.rerun()

    st.markdown("### Lista de Objetivos")
    
    if not alloc_df.empty:
        alloc_map = alloc_df.groupby("objective_id")["amount"].sum().to_dict()
    else:
        alloc_map = {}

    f_floor = st.selectbox("Ver Andar:", ["Todos"] + FLOOR_OPTIONS)
    view_objs = obj_df if f_floor == "Todos" else obj_df[obj_df["floor"] == f_floor]
    
    if view_objs.empty:
        st.info("Sem objetivos cadastrados.")
    else:
        for _, row in view_objs.iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 4, 2, 1])
                saved_amt = alloc_map.get(str(row["id"]), 0.0)
                # Safety Division
                pct = (saved_amt / row["target"]) if row["target"] > 0 else 0
                
                with c1:
                    st.markdown(f"**{row['name']}**")
                    st.caption(f"Prio: {row['priority']} • {row['floor']}")
                with c2: 
                    st.progress(min(pct, 1.0))
                    st.caption(f"{pct*100:.1f}%")
                with c3: st.markdown(f"{format_currency(saved_amt)} / {format_currency(row['target'])}")
                with c4:
                    if st.button("🖊️", key=f"btn_edit_{row['id']}"):
                        st.session_state.edit_obj_id = str(row["id"])
                        st.rerun()

# ==============================================================================
# 7. PAGE: ALLOCATIONS
# ==============================================================================
elif page == "🧱 Alocações":
    st.title("🧱 Alocar Dinheiro")
    
    if tx_df.empty or obj_df.empty:
        st.warning("Você precisa ter Transações (Entradas) e Objetivos cadastrados.")
    else:
        income_txs = tx_df[tx_df["type"] == "income"].copy()
        
        # Mapping for safe selection: {Label: ID}
        inc_options = {
            f"{x['date'].strftime('%d/%m')} - {x['category']} - {format_currency(x['amount'])}": x['id'] 
            for _, x in income_txs.iterrows()
        }
        
        sel_label = st.selectbox("Selecione a Entrada de Dinheiro:", list(inc_options.keys()))
        
        if sel_label:
            sel_id = inc_options[sel_label]
            sel_tx = income_txs[income_txs["id"] == sel_id].iloc[0]
            
            st.info(f"💲 Valor Total da Entrada: **{format_currency(sel_tx['amount'])}**")
            
            existing_allocs = alloc_df[alloc_df["tx_id"] == str(sel_id)]
            used = existing_allocs["amount"].sum()
            available = sel_tx["amount"] - used
            
            col_metrics1, col_metrics2 = st.columns(2)
            col_metrics1.metric("Já Alocado", format_currency(used))
            col_metrics2.metric("Disponível", format_currency(available), delta_color="normal" if available >= 0 else "inverse")
            
            st.markdown("---")
            
            if available > 0:
                with st.form("make_alloc"):
                    st.write("↳ **Destinar para:**")
                    
                    # Mapping for safe objective selection
                    obj_options = {
                        f"{x['name']} ({x['floor']}) - Meta: {format_currency(x['target'])}": x['id']
                        for _, x in obj_df.iterrows()
                    }
                    
                    sel_obj_label = st.selectbox("Objetivo", list(obj_options.keys()))
                    amount_to_alloc = st.number_input("Quanto alocar?", min_value=0.0, max_value=float(available), step=50.0)
                    
                    if st.form_submit_button("✔️ Confirmar Alocação"):
                        if amount_to_alloc > 0:
                            target_obj_id = obj_options[sel_obj_label]
                            new_alloc = {
                                "tx_id": str(sel_id),
                                "objective_id": str(target_obj_id),
                                "amount": amount_to_alloc
                            }
                            alloc_df = pd.concat([alloc_df, pd.DataFrame([new_alloc])], ignore_index=True)
                            save_data("alloc", alloc_df)
                            st.success("Dinheiro alocado com sucesso!")
                            st.rerun()
            else:
                st.success("✔️ Todo o valor desta entrada já foi distribuído!")

            if not existing_allocs.empty:
                st.markdown("#### 📄 Distribuição Atual:")
                history = existing_allocs.merge(obj_df[["id", "name", "floor"]], left_on="objective_id", right_on="id")
                history_display = format_df_br(history, ["amount"])
                st.dataframe(history_display[["name", "floor", "amount"]], use_container_width=True)
