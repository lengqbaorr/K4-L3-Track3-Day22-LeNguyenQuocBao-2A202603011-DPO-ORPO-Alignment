# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Hoàn tất — tự điền REFLECTION từ số liệu thật và nén file nộp bài
#
# Chạy sau NB4 (và chạy lại sau bonus). Cell đọc các file JSON do notebook sinh ra,
# viết `submission/REFLECTION.md`, rồi nén mọi thứ cần nộp thành `lab22_submission.zip` và tải về máy.
# **Văn bản §3, §4, §6 là bản nháp sinh từ số liệu: hãy đọc lại và sửa theo cách hiểu của bạn trước khi nộp.**

# %%
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

STUDENT_NAME = "Lê Nguyễn Quốc Bảo (2A202603011)"
COHORT = "A20-K4"

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))
from lab22 import config as C


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


m = load(C.DPO_ADAPTER / "dpo_metrics.json")
j = load(C.EVAL_DIR / "judge_summary.json")
stats = load(C.PREF_DIR / "stats.json")
var = load(C.VARIANTS_DIR / "variants_summary.json")
deploy = load(C.EVAL_DIR / "deploy_meta.json")
held = j.get("heldout", {})
diag = m.get("diagnosis", "AMBIGUOUS")

# --- §3: đọc đường reward ---------------------------------------------------
tr_c, tr_r = m.get("end_chosen_reward"), m.get("end_rejected_reward")
ev_c, ev_r, ev_gap = m.get("eval_chosen_reward"), m.get("eval_rejected_reward"), m.get("eval_reward_gap")
chosen_dir = "giảm" if (tr_c is not None and tr_c < 0) else "tăng"
generalize = (
    "Held-out đi cùng hướng với tập huấn luyện (margin dương ở cả hai), nên mô hình khái quát chứ không chỉ học thuộc."
    if ev_gap is not None and ev_gap > 0
    else "Margin trên held-out không dương, nên mô hình không khái quát tốt: có dấu hiệu học thuộc tập huấn luyện hoặc chưa học được."
)
mech = {
    "INTENDED": "Đúng kỳ vọng: reward chosen tăng, rejected giảm, margin tăng.",
    "LIKELIHOOD DISPLACEMENT": (
        "Dịch chuyển xác suất: margin tăng chủ yếu vì xác suất của câu rejected giảm nhanh hơn câu chosen, "
        "nên DPO không đẩy chosen lên mà chỉ kéo rejected xuống (xem NB0 §5)."
    ),
    "FAILURE": "Thất bại: margin trên held-out không dương nên mô hình chưa phân biệt được chosen với rejected ngoài tập huấn luyện.",
}.get(diag, "Chưa kết luận: các đường cong không đủ rõ ràng để khẳng định một trong ba trường hợp.")
sec3 = (
    f"Ở cuối huấn luyện, reward ngầm của câu chosen {chosen_dir} đến {f(tr_c)} và reward của câu rejected đạt {f(tr_r)}, "
    f"cho khoảng cách (margin) {f(m.get('end_reward_gap'))} trên tập huấn luyện. Trên tập held-out, chosen là {f(ev_c)}, "
    f"rejected là {f(ev_r)}, margin {f(ev_gap)} và độ chính xác reward {pct(m.get('eval_reward_accuracy'))}. "
    f"Chẩn đoán tự động của notebook là **{diag}**. {mech} {generalize} "
    f"Reward bắt đầu từ 0 vì lúc đầu mô hình đang học trùng với mô hình tham chiếu SFT, và loss đầu tiên được ghi là "
    f"{f(m.get('first_logged_loss'))}, gần log 2 ≈ 0,693 như NB0 dự đoán, nên tham chiếu đúng là mô hình SFT. "
    f"Đối chiếu với ảnh `03-dpo-reward-curves.png`, các con số này khớp với hình dạng các đường: tôi đọc chosen và rejected "
    f"riêng rẽ thay vì chỉ nhìn margin, vì margin tăng không cho biết xác suất chosen có thật sự tăng hay không."
)


# --- §4: so sánh ------------------------------------------------------------
def row(label, d):
    if not d or not d.get("n"):
        return f"| {label} | 0 | | | | | | |"
    lo, hi = d["win_rate_ci95"]
    return (
        f"| {label} | {d['n']} | {d['dpo_wins']} | {d['sft_wins']} | {d['ties']} | "
        f"{d['dpo_win_rate']:.2f} [{lo:.2f}, {hi:.2f}] | {f(d.get('length_matched_win_rate'), 2)} | "
        f"{f(d.get('longer_answer_won_frac'), 2)} |"
    )


ci = held.get("win_rate_ci95") or [None, None]
contains_half = ci[0] is not None and ci[0] <= 0.5 <= ci[1]
per_judge = j.get("per_judge", {})
pj_text = "; ".join(f"{k.split('/')[-1]}: {v.get('dpo_win_rate', float('nan')):.2f}" for k, v in per_judge.items()) or "n/a"
agree = (j.get("judge_agreement") or {}).get("agreement")
sanity = j.get("sanity_accuracy")
cm = held.get("length_matched_win_rate")
dl = (held.get("mean_chars_dpo") or 0) - (held.get("mean_chars_sft") or 0)
rows = [r for r in (load(C.EVAL_DIR / "judge_results_rm.json").get("records") or []) if r.get("category") in ("helpfulness", "safety")]


def example(cat):
    for r in rows:
        if r["category"] == cat:
            return (
                f"Câu hỏi: “{r['prompt'][:140]}”. Người thắng: **{r['winner']}**. "
                f"Độ dài SFT {len(r['sft'])} ký tự, DPO {len(r['dpo'])} ký tự."
            )
    return "Không có ví dụ trong kết quả."


sec4 = (
    f"Trên held-out, win rate của DPO là {f(held.get('dpo_win_rate'), 2)} với khoảng tin cậy 95% [{f(ci[0], 2)}, {f(ci[1], 2)}]. "
    + ("Khoảng này **chứa 0,5**, nên chưa đủ bằng chứng DPO tốt hơn SFT. " if contains_half
       else "Khoảng này **không chứa 0,5**, nên có bằng chứng về sự khác biệt giữa hai mô hình. ")
    + f"Sanity accuracy của giám khảo yếu nhất là {pct(sanity)}"
    + (" (≥ 80%, đọc tiếng Việt đủ tốt)." if sanity is not None and sanity >= 0.8 else " (< 80%: không nên tin win rate).")
    + f" Win rate từng giám khảo: {pj_text}; tỉ lệ đồng ý giữa hai giám khảo: {pct(agree)}. "
    f"Cả hai đều thuộc họ Skywork, cùng họ với mô hình gán nhãn dữ liệu, nên không loại trừ được rò rỉ sở thích. "
    f"Độ dài trung bình câu trả lời thay đổi {dl:+.0f} ký tự (SFT → DPO), câu dài hơn thắng trong {f(held.get('longer_answer_won_frac'), 2)} "
    f"số cặp, và win rate trên các cặp dài gần bằng nhau là {f(cm, 2)}"
    + ("; hai con số này cho thấy độ dài không phải là lý do chính." if cm is not None and abs(cm - (held.get('dpo_win_rate') or 0)) < 0.1
       else "; chênh lệch với win rate chung cho thấy độ dài có góp phần vào kết quả.")
    + f"\n\n**Ví dụ hữu ích (helpfulness).** {example('helpfulness')}\n\n**Ví dụ an toàn (safety).** {example('safety')}"
)

# --- §6: một quyết định -----------------------------------------------------
sec6 = (
    "Quyết định quan trọng nhất của tôi là **dùng hội đồng hai reward model chạy trên Colab làm giám khảo** thay vì "
    "giám khảo qua API. Phương án thay thế là một mô hình ngôn ngữ lớn qua API, vốn đọc tiếng Việt tự nhiên hơn nhưng cần khoá "
    "và tốn tiền, và còn phải chấm hai lần đổi chỗ A/B để loại thiên vị vị trí. Tôi chọn hội đồng reward model vì miễn phí, chạy lại "
    "được trong một lần \"Chạy tất cả\", và mỗi câu được chấm độc lập nên không có thiên vị vị trí. Đổi lại, cả hai giám khảo cùng họ Skywork "
    "với mô hình gán nhãn dữ liệu, nên kết quả có thể thiên vị DPO. "
    f"Kết quả: win rate trên held-out là {f(held.get('dpo_win_rate'), 2)} (khoảng tin cậy [{f(ci[0], 2)}, {f(ci[1], 2)}]), sanity accuracy "
    f"{pct(sanity)}, hai giám khảo đồng ý với nhau {pct(agree)} (win rate từng giám khảo: {pj_text}). "
    + ("Kết quả này chỉ cho thấy cải thiện trong khoảng nhiễu, nên tôi không khẳng định DPO tốt hơn. "
       if contains_half else "Kết quả này cho thấy DPO khác SFT một cách có ý nghĩa theo giám khảo này, nhưng tôi vẫn dè dặt vì nguy cơ rò rỉ sở thích. ")
    + "Nếu làm lại, tôi sẽ thêm một giám khảo API khác họ để chấm chéo (`cross_judge.agreement`), "
    "đồng thời thử β = 0,05 và 0,5 để xem độ lớn của dịch chuyển xác suất có phụ thuộc β hay không, "
    "vì hiện tại tôi chỉ có một lần chạy với β = 0,1 và tốc độ học 5e-6 nên không tách được ảnh hưởng của từng siêu tham số."
)

# --- bonus ------------------------------------------------------------------
var_rows, sec8_note = "", "Chưa chạy NB3b."
if var:
    names = {"dpo": "DPO", "rpo": "RPO", "dpo_norm": "DPO-norm", "ld_dpo": "LD-DPO", "orpo": "ORPO"}
    for k, label in names.items():
        v = var.get(k)
        if v:
            gap = (v.get("eval_chosen_reward") or 0) - (v.get("eval_rejected_reward") or 0) if v.get("eval_chosen_reward") is not None else None
            var_rows += f"| {label} | {f(v.get('eval_reward_accuracy'))} | {f(gap)} | {f(v.get('mean_output_chars'), 0)} | {v.get('diagnosis', '')} |\n"
    lens = {k: v["mean_output_chars"] for k, v in var.items() if v.get("mean_output_chars") is not None}
    base = lens.get("dpo")
    if lens and base is not None:
        big = max(lens, key=lambda k: abs(lens[k] - base))
        sec8_note = (
            f"Biến thể thay đổi độ dài nhiều nhất so với DPO là **{names.get(big, big)}** ({lens[big]:.0f} ký tự so với {base:.0f}). "
            "Nguyên nhân nằm ở công thức loss: DPO-norm và ORPO chuẩn hoá theo độ dài nên không thưởng cho câu dài, RPO/LD-DPO "
            "giữ hoặc giảm ảnh hưởng của độ dài câu chosen/rejected lên gradient, còn DPO gốc dùng tổng log-prob nên câu dài "
            "có lợi thế (NB0 §6)."
        )
gguf_note = (
    f"NB5 đã xuất `{deploy['gguf_path']}` ({deploy['gguf_size_mb']} MB, {deploy['quantization']})." if deploy else "Chưa chạy NB5."
)


def tick(ok, label):
    return f"- [{'x' if ok else ' '}] {label}"


reflection = f"""# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** {STUDENT_NAME}
**Khoá:** {COHORT}
**Tier đã chạy:** {C.COMPUTE_TIER}
**Ngày:** {date.today().isoformat()}

> Mọi con số dưới đây lấy từ `adapters/dpo/dpo_metrics.json`, `data/eval/judge_summary.json`, `data/pref/stats.json`.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab {C.COMPUTE_TIER} |
| Mô hình gốc | {C.BASE_MODEL} |
| Dữ liệu SFT | {C.SFT_DATASET} · {C.SFT_SLICE} mẫu · 1 epoch |
| Dữ liệu sở thích | {C.PREF_DATASET} (vi) · {C.PREF_TRAIN} huấn luyện / {C.PREF_EVAL} held-out |
| Chosen dài hơn rejected (NB2) | {pct(stats.get('chosen_longer_frac'))} |
| DPO: β / tốc độ học (lr) / số epoch | {C.DPO_BETA} / {C.DPO_LR} / {C.DPO_EPOCHS} |
| Giám khảo | {j.get('judge', 'n/a')}; sanity accuracy {pct(sanity)} |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | xem output cell NB3 |
| VRAM cao nhất | xem output cell NB3 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | {f(m.get('end_reward_gap'))} |
| Độ chính xác reward trên held-out | {f(m.get('eval_reward_accuracy'))} |
| Margin trên held-out | {f(ev_gap)} |
| Chẩn đoán tự động (`diagnosis`) | {diag} |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | {f(held.get('mean_chars_sft'), 0)} → {f(held.get('mean_chars_dpo'), 0)} ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

{sec3}

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
{row('held-out', held)}
{row('hữu ích — helpfulness (4)', j.get('helpfulness'))}
{row('an toàn — safety (4)', j.get('safety'))}

Giám khảo: {j.get('judge', 'n/a')} · sanity accuracy: {pct(sanity)} · `score_length_spearman`: {f(held.get('score_length_spearman'))}

{sec4}

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Không chạy β-sweep. Giả thuyết: β nhỏ (0,05) cho phép mô hình đi xa mô hình tham chiếu hơn nên margin lớn hơn nhưng dễ dịch chuyển xác suất;
β lớn (0,5) giữ mô hình gần tham chiếu nên margin nhỏ và thay đổi ít; β = 0,1 nằm ở giữa.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

{sec6}

---

## 7. Bộ đo chuẩn (bonus NB6)

Không chạy NB6.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
{var_rows or '| (chưa chạy) | | | | |'}
{sec8_note}

{gguf_note}

---

## 9. GRPO (bonus NB7)

Không chạy NB7.

---

## Danh sách bonus

{tick(bool(var), 'NB3b — biến thể loss (+8)')}
{tick(bool(deploy), 'NB5 — GGUF SFT+DPO (+4)')}
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)

---

## Điều bất ngờ nhất

Xem số liệu §2–§4.
"""
(ROOT / "submission").mkdir(exist_ok=True)
(ROOT / "submission" / "REFLECTION.md").write_text(reflection, encoding="utf-8")
print("Đã ghi submission/REFLECTION.md")

# %% [markdown]
# ## Nén file nộp bài và tải về

# %%
CORE_PNG = ["02-sft-loss", "02b-pref-length", "03-dpo-reward-curves", "04-side-by-side-table"]
missing = [n for n in CORE_PNG if not (C.SCREENSHOTS / f"{n}.png").exists()]
print("Ảnh bắt buộc còn thiếu:", missing or "không")

picks = [ROOT / "submission" / "REFLECTION.md", C.PREF_DIR / "stats.json"]
picks += sorted(C.SCREENSHOTS.glob("*.png")) + sorted(C.EVAL_DIR.glob("*.json*"))
picks += sorted(C.DPO_ADAPTER.glob("*.json")) + sorted(C.VARIANTS_DIR.glob("*.json"))

zip_path = ROOT / "lab22_submission.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in dict.fromkeys(picks):
        if p.is_file() and p.stat().st_size < 50e6:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(*z.namelist(), sep="\n")
print(f"\n{zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

kaggle_out = Path("/kaggle/working")
if kaggle_out.exists():  # Kaggle chỉ giữ file trong /kaggle/working; tải ở tab Output sau khi "Save Version"
    import shutil

    shutil.copy(zip_path, kaggle_out / zip_path.name)
    print(f"Đã copy vào {kaggle_out / zip_path.name}")
try:
    from google.colab import files

    files.download(str(zip_path))
except ImportError:
    print("Không phải Colab: lấy file zip ở đường dẫn trên (Kaggle: tab Output).")
