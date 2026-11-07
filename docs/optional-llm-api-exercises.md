# Ngày 1 — Bài Tập & Phản Ánh (side lab TUỲ CHỌN)

> Side lab này **không** thuộc điểm chính của Day 01 (xem [RUBRIC.md](../RUBRIC.md) mục Bonus). Model: `gpt-6.1-terra` (lớn) và `gpt-6-luna` (nhỏ). `gpt-4o` / `gpt-4o-mini` bị gỡ khỏi API ngày 14/01/2027 ([nguồn](https://developers.openai.com/api/docs/deprecations)). Tên khoá `gpt4o_*` / `mini_*` trong kết quả là hợp đồng của test, giữ nguyên để tương thích: đọc là "model lớn" / "model nhỏ".
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Copy `src/optional_llm_api/template.py` thành `src/optional_llm_api/solution/solution.py`, triển khai mọi TODO, rồi chạy từ gốc repo: `pytest tests/test_solution.py -v`. Chưa có `solution/solution.py` thì test tự bỏ qua.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python src/optional_llm_api/solution/solution.py
```
Bạn sẽ thấy output so sánh phản hồi của `gpt-6.1-terra` và `gpt-6-luna`.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."** (nhớ gửi `reasoning_effort="none"`: model GPT-6 chỉ nhận `temperature` khi tắt suy luận)

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Kết quả chạy thật với `gpt-6.1-terra` (`reasoning_effort="none"`, `max_tokens=256`):
>
> | temperature | Nội dung chính | Độ dài |
> |---|---|---|
> | 0.0 | Việt Nam xuất khẩu hạt điều lớn nhất thế giới, phần lớn hạt điều thô nhập từ châu Phi | 40 từ |
> | 0.5 | Hang Sơn Đoòng lớn nhất thế giới, bên trong có rừng và hệ sinh thái riêng | 34 từ |
> | 1.0 | Hang Sơn Đoòng (gần như cùng câu với 0.5, chỉ đổi vài chữ) | 32 từ |
> | 1.5 | Hang Sơn Đoòng ở Quảng Bình, thêm chi tiết ánh sáng lọt qua hố sụt trên trần hang | 33 từ |
>
> Ở temperature 0, model chọn câu trả lời "chắc nhất" theo xác suất, nên kết quả cố định và lặp lại được. Khi tăng temperature, model được phép chọn token ít khả năng hơn, nên đổi sang một sự thật khác và cách diễn đạt tự do hơn (ở 1.5 có thêm chi tiết riêng). Tuy vậy, với prompt ngắn và `top_p=0.9`, ba mức 0.5–1.5 vẫn cùng xoay quanh Sơn Đoòng: temperature làm câu chữ đa dạng hơn chứ chưa chắc đổi ý chính, và mỗi mức chỉ chạy một lần nên chưa đủ để kết luận chắc chắn.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Khoảng **0.0–0.3**. Chatbot hỗ trợ khách hàng cần trả lời đúng, nhất quán và lặp lại được: cùng một câu hỏi về chính sách đổi trả thì hai khách phải nhận cùng một câu trả lời, và khi có sự cố thì dễ tái hiện để sửa. Temperature cao làm tăng rủi ro model "sáng tạo" ra thông tin sai (chính sách, giá, thời hạn). Có thể để hơi lớn hơn 0 (khoảng 0.2–0.3) để câu chữ đỡ máy móc, nhưng nội dung thì vẫn phải dựa trên tài liệu chính thức.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem `gpt-6.1-terra` đắt hơn `gpt-6-luna` bao nhiêu lần cho workload này** (giá 11/2026: terra $1.00 / $5.00, luna $0.10 / $0.50 mỗi 1M token input / output):
> Khối lượng: 10 000 người × 3 lần gọi = **30 000 lần gọi/ngày**, × 350 token = **10,5 triệu token/ngày**.
>
> Đề không nói tỉ lệ input/output, nên giả sử mỗi lần gọi có 250 token input và 100 token output:
>
> | | Input (7,5M token) | Output (3M token) | Tổng/ngày | ~30 ngày |
> |---|---|---|---|---|
> | `gpt-6.1-terra` | 7,5 × $1.00 = $7.50 | 3 × $5.00 = $15.00 | **$22.50** | ~$675 |
> | `gpt-6-luna` | 7,5 × $0.10 = $0.75 | 3 × $0.50 = $1.50 | **$2.25** | ~$67.5 |
>
> → `gpt-6.1-terra` đắt hơn **10 lần**. Con số 10 lần này không phụ thuộc vào giả định tỉ lệ input/output, vì cả giá input lẫn giá output của terra đều gấp đúng 10 lần luna. Tỉ lệ input/output chỉ thay đổi số tiền tuyệt đối (ví dụ nếu toàn bộ 350 token là output thì terra tốn $52.50/ngày, luna $5.25/ngày).

**Mô tả một trường hợp mà chi phí cao hơn của `gpt-6.1-terra` là xứng đáng, và một trường hợp `gpt-6-luna` là lựa chọn tốt hơn:**
> - **Terra xứng đáng:** khi một câu trả lời sai gây thiệt hại lớn hơn nhiều so với tiền API, và số lượng gọi ít. Ví dụ trợ lý phân tích hợp đồng hoặc tóm tắt hồ sơ y tế cho bác sĩ: mỗi ngày chỉ vài trăm lần gọi, cần suy luận nhiều bước và đọc văn bản dài; trả thêm vài đô mỗi ngày là rẻ so với rủi ro hiểu sai một điều khoản.
> - **Luna tốt hơn:** khi tác vụ đơn giản, lặp lại và khối lượng lớn, mà model nhỏ đã làm đủ tốt. Ví dụ phân loại spam/ham ở lab chính: trên 40 tin SUB, `gpt-6-luna` và `gpt-6.1-terra` đều đạt F1 = 1.0, nhưng terra đắt gấp khoảng 10 lần ($0.091 so với $0.009 cho 1 000 tin) và còn chậm hơn ở p50. Với hàng triệu tin mỗi ngày, chọn terra chỉ tốn thêm tiền mà không tăng chất lượng.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi **có người đang ngồi chờ đọc một câu trả lời dài**, như chatbot, trợ lý viết hoặc giải thích code. Tổng thời gian sinh xong câu trả lời không giảm, nhưng chữ đầu tiên xuất hiện sau khoảng một giây thay vì phải đợi cả đoạn, nên người dùng thấy hệ thống phản hồi ngay, có thể đọc dần và bấm dừng nếu câu trả lời đi sai hướng. Ngược lại, non-streaming phù hợp hơn khi **output được máy đọc chứ không phải người đọc**, hoặc câu trả lời rất ngắn: phân loại spam/ham chỉ trả về một từ, trích xuất JSON cần đủ cả khối mới parse được, các job chạy nền theo lô, hoặc khi cần kiểm duyệt toàn bộ nội dung trước khi hiển thị. Trong những trường hợp đó, streaming chỉ làm code phức tạp hơn (ghép từng mảnh, xử lý lỗi giữa chừng) mà không mang lại lợi ích gì cho trải nghiệm.


## Danh Sách Kiểm Tra Nộp Bài
- [ ] Tất cả tests pass: `pytest tests/test_solution.py -v`
- [ ] `call_openai` đã triển khai và kiểm thử
- [ ] `call_openai_mini` đã triển khai và kiểm thử
- [ ] `compare_models` đã triển khai và kiểm thử
- [ ] `streaming_chatbot` đã triển khai và kiểm thử
- [ ] `retry_with_backoff` đã triển khai và kiểm thử
- [ ] `batch_compare` đã triển khai và kiểm thử
- [ ] `format_comparison_table` đã triển khai và kiểm thử
- [ ] `exercises.md` đã điền đầy đủ
- [ ] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
