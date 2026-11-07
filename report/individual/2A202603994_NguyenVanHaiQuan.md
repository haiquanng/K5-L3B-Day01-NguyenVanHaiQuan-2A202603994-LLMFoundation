# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV: 2A202603994**
- **Họ và tên: Nguyễn Văn Hải Quân**
- **SEED** (in ở ô đầu notebook): 5234

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output |  ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra |  ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước |  ✅ |

## 2. Kết quả chính

| Cách tiếp cận | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 0.843 | 0.889 | 0.014 | 0 |
| 2. BoW + MLP | 0.911 | 1.000 | 0.039 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.905 | 0.976 | 16.3 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.931 | 1.000 | 5.4 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.929 | 1.000 | 5.3 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | — | 1.000 | 1 940 | 0.009 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | — | 1.000 | 1 534 | 0.091 |
| 4c. LLM nhỏ few-shot | — | 1.000 | 1 236 | 0.021 |

LLM chỉ chạy trên tập SUB (40 tin) để tiết kiệm chi phí, nên không có F1 trên toàn bộ test. Trên SUB, 6/8 cách đạt F1 = 1.0, nên SUB gần như không phân biệt được các cách với nhau. Nhìn F1 trên toàn bộ test thì Encoder + LogReg tốt nhất (0.931). Đường Pareto (1 − F1 theo độ trễ) chỉ gồm BoW + LogReg và BoW + MLP. Có 5 cách đạt SLA p95 < 50 ms: cả hai cách BoW và cả ba cách Encoder (3a, 3b, 3c); cả ba cách LLM đều không đạt.

## 3. Dự đoán và thực tế

**`ml_lr` — learning rate = 50.** Mình đoán loss sẽ dao động răng cưa và không hội tụ. Thực tế loss chỉ vọt lên khoảng 2.5 ở epoch đầu rồi giảm rất nhanh về 0.005, còn thấp hơn cả lr = 0.5. Lý do: loss của logistic regression có dạng "cái bát" chỉ có một đáy, nên dù bước quá đà thì gradient vẫn kéo về đúng hướng; feature lại chỉ là 0/1 nên mỗi bước không văng quá xa.

**`emb_tricky` — 6 tin "ngụy trang".** Mình đoán Embedding sẽ thắng rõ (khoảng 5/6 so với 3/6). Thực tế BoW đúng 5/6, Embedding chỉ 4/6. BoW không bị chữ "free", "won", "prize" lừa như mình nghĩ, vì các từ rất "ham" như *I, me, my, lol* kéo xác suất xuống. Ngược lại, tin "complimentary holiday… Reply YES" thì BoW vẫn bắt được nhờ các từ *mobile, reply, yes*, còn Embedding lại để lọt. Cả hai cùng sai tin phishing "Your account rewards expire tonight", vì kiểu lừa đảo này gần như không có trong dữ liệu train.

**`llm_models` — LLM nhỏ và lớn.** Mình đoán model lớn hơn model nhỏ một chút. Lần chạy đầu, cả 3 cách dùng LLM đều ra F1 = 0 với invalid = 40/40, vì `max_tokens = 3` quá ít khiến model trả về chuỗi rỗng. Mình kiểm tra thấy API vẫn chạy (ô "Say hello" với 20 token trả lời bình thường) và chi phí gần như chỉ có phần input, nên đổi `max_tokens` lên 20. Sau khi sửa, cả model nhỏ lẫn model lớn đều đạt F1 = 1.0, tức hoà nhau, trong khi model lớn đắt gấp khoảng 10 lần.

## 4. Phân tích 50 case khó

| Model | F1 SUB 40 | Accuracy 50 case khó |
|---|---|---|
| 1. BoW + LogReg | 0.889 | 54% |
| 2. BoW + MLP | 1.000 | 50% |
| 3a. Emb + kNN | 0.976 | 70% |
| 3b. Emb + LogReg | 1.000 | 60% |
| 3c. Emb + MLP head | 1.000 | 68% |
| 4a / 4d. LLM nhỏ (thường / prompt phòng thủ) | 1.000 | 100% |

Tụt nhiều nhất là **BoW + MLP**: từ F1 = 1.0 trên SUB xuống 50% trên bộ case khó, ngang đoán bừa. Nó đúng 0% ở hai nhóm *spam giọng tự nhiên* và *né từ khoá*. Hai case cụ thể:

- **#29** *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…"* (né từ khoá): khi thay chữ bằng số, các từ quen thuộc như "congratulations", "won", "gift" biến thành token lạ, không có trong vocab 2000 từ. Vector đầu vào gần như toàn 0, chỉ còn bias kéo về ham.
- **#14** *"Hi, I'm working from home and making about £300 a day, happy to show you how…"* (spam giọng tự nhiên): câu viết như người quen kể chuyện, gồm toàn từ bình thường (*I, home, happy, you, me*). Những từ này kéo mạnh về phía ham, nên BoW không nhận ra đây là lừa việc làm.

Embedding chịu được né từ khoá tốt hơn (83%) nhưng yếu ở *spam giọng tự nhiên* (10–30%) và *nói về spam* (25–50%), vì nó nhìn nghĩa chung của câu chứ không phân biệt được ai đang nói và nói để làm gì. LLM đúng 100% mọi nhóm, kể cả prompt injection, nên prompt phòng thủ (4d) không cải thiện thêm nhưng cũng không làm hỏng nhóm nào. Điều này cho thấy điểm cao trên tập test thường là quá lạc quan, vì test lấy cùng nguồn với train và chủ yếu là spam kiểu cũ.

## 5. Thử thách cá nhân

**Đề #3:** DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?

**Cách làm:** Dùng lại `build_mlp` và `train_mlp` của Phần 2, chạy vòng lặp với `hidden` = 4, 64 (mốc so sánh) và 512. Giữ cùng 30 epoch và cùng `SEED` để chỉ có kích thước lớp ẩn thay đổi. Đo thời gian bằng `time.perf_counter()`, đếm tham số bằng `p.numel()`, và kiểm tra lại bằng công thức 2000·h + 2h + 1.

**Kết quả:**

| Lớp ẩn | Tham số | Precision | Recall | F1 test | Train (s) |
|---|---|---|---|---|---|
| 4 | 8 009 | 0.978 | 0.854 | 0.912 | 0.54 |
| 64 | 128 129 | 0.989 | 0.845 | 0.911 | 0.64 |
| 512 | 1 025 025 | 1.000 | 0.845 | 0.916 | 3.69 |

**Giải thích:** Tăng từ 4 lên 512 nơ-ron làm số tham số gấp ~128 lần và thời gian train gấp ~7 lần, nhưng F1 gần như không đổi (0.912 → 0.916, chênh khoảng 1 tin). Với input bag-of-words, vài từ khoá đã đủ để tách spam và ham, nên mạng rất nhỏ đã học được gần hết những gì có thể học. Nên chọn mạng nhỏ khi nó cho chất lượng gần bằng mạng lớn, khi dữ liệu ít (dễ overfit), hoặc khi cần chạy nhanh, rẻ, ít bộ nhớ.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây), p95 < 50 ms/tin.

**Phương án chọn: Encoder (MiniLM) + LogReg (3b).**
- Đạt SLA thoải mái: p95 = 5.4 ms.
- F1 trên toàn bộ test cao nhất (0.931).
- Xử lý được khoảng 1 237 tin/giây trên một máy, gấp hơn 10 lần lưu lượng trung bình, nên vài máy là đủ chịu giờ cao điểm.
- Không tốn tiền API.

**Vì sao không chọn các cách khác:**
- BoW + MLP nhanh hơn (p95 0.04 ms) nhưng F1 thấp hơn (0.911) và rất dễ bị lừa (50% trên case khó).
- Encoder + kNN cũng đạt SLA (p95 = 16.3 ms) nhưng F1 thấp hơn (0.905), chậm gấp khoảng 3 lần 3b và chỉ xử lý được khoảng 346 tin/giây, tức chỉ gấp khoảng 3 lần lưu lượng trung bình, ít dư cho giờ cao điểm.
- LLM chính xác nhất (100% trên case khó) nhưng p95 khoảng 1.2–1.9 giây, chậm hơn SLA khoảng 25–39 lần. Nếu chạy cho toàn bộ 10 triệu tin bằng model nhỏ zero-shot thì tốn khoảng $90/ngày, tức khoảng $2 700/tháng; few-shot khoảng $6 200/tháng; model lớn khoảng $27 000/tháng.

**Đánh đổi chấp nhận:** 3b chỉ đúng 60% trên bộ case khó, nghĩa là spam giọng tự nhiên và lừa đảo kiểu mới vẫn lọt. Để bù lại, có thể thêm một lớp **router**: 3b xử lý trực tiếp những tin nó chắc chắn, còn tin "lưỡng lự" (xác suất 0.3–0.7, chỉ chiếm phần nhỏ) được gửi sang LLM kiểm tra lại *sau khi đã giao tin* (không chặn trên đường đi), hoặc dùng để gán nhãn thêm dữ liệu rồi train lại 3b định kỳ.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Câu hỏi cá nhân (mục 3, 5, 6) | Hỏi AI tra cứu thông tin để trả lời các câu hỏi cá nhân | Giải thích khái niệm và gợi ý hướng trả lời | Đối chiếu với kết quả chạy notebook, viết lại câu trả lời theo ý mình |
| Các TODO trong notebook | Hỏi về thuật toán (BoW, MLP, embedding, gọi LLM) để xem code mẫu | Đưa ra code mẫu và giải thích cách hoạt động | Chỉnh code cho khớp với dữ liệu của lab, chạy lại notebook để kiểm tra kết quả |
| Báo cáo này | Nhờ AI tạo file report dựa trên kết quả của notebook | Viết bản nháp báo cáo theo template | Kiểm tra lại các con số (F1, độ trễ, chi phí) với output notebook và sửa phần phân tích |
| Rà soát bài và cập nhật số liệu | Nhờ AI rà các phần còn thiếu, đối chiếu số liệu câu trả lời với output lần chạy mới, cập nhật báo cáo theo số mới | Danh sách chỗ thiếu / sai số liệu; cập nhật bảng kết quả (mục 2) và phương án sản phẩm (mục 6) | Tự sửa các câu trả lời trong notebook; kiểm tra lại số trong báo cáo với output notebook |
| Bonus: side lab LLM API | Nhờ AI chạy bài tập temperature (gọi API thật) và soạn câu trả lời cho `docs/optional-llm-api-exercises.md` | Kết quả 4 mức temperature, phép tính chi phí terra/luna, đoạn văn về streaming | Đọc lại, đối chiếu với output thật và kết quả lab chính; `solution.py` chạy qua 19/19 test |
| Bonus: `MY_CASES` | Case do mình tự nghĩ; nhờ AI soạn nhận xét từ bảng kết quả từng case | Bảng tổng hợp + nhận xét model nào bị lừa ở case nào và vì sao | Đối chiếu với bảng True/False của notebook, chỉnh lại câu chữ |

**Điều mình hiểu ra hôm nay:**
1. Điểm cao trên tập test chưa chắc model đã hiểu bài toán. BoW + MLP đạt F1 = 1.0 trên SUB nhưng chỉ đúng 50% khi gặp tin cố tình đánh lừa.
2. Hệ thống dùng LLM có thể hỏng mà không báo lỗi: chỉ vì `max_tokens = 3`, cả model lớn nhất cũng ra F1 = 0, code vẫn chạy và vẫn tốn tiền. Phải luôn theo dõi tỉ lệ invalid.
3. Chọn model cho sản phẩm thật là đánh đổi giữa chất lượng, độ trễ và chi phí, không phải cứ chọn model mạnh nhất. LLM tốt nhất về chất lượng nhưng không đáp ứng được 50 ms.

