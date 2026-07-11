## **Giải thích đề bài: Hệ thống Multi-Agent LLM cho MedQA-USMLE**

Đây là một đồ án giữa kỳ (midterm project) yêu cầu bạn xây dựng một **hệ thống nhiều AI agent phối hợp với nhau** để trả lời câu hỏi trắc nghiệm y khoa (MedQA-USMLE — bộ đề thi cấp phép hành nghề y ở Mỹ), với mục tiêu chính là **tăng độ chính xác** so với việc chỉ hỏi thẳng một LLM.

Dưới đây là giải thích từng phần:

### **1\. Mục tiêu & phạm vi**

* Bạn không cần xây hệ thống để *dùng thật trong lâm sàng* — chỉ cần đánh giá **thiết kế hệ thống** (kiến trúc, hiệu quả) trên benchmark MedQA-USMLE.  
* Input: 1 câu hỏi trắc nghiệm y khoa → Output: 1 đáp án \+ giải thích ngắn.

### **2\. Các thành phần bắt buộc phải có**

Bạn phải xây **7 thành phần**:

1. **\[Đạt nghiên cứu được API \+ Huy sẽ thực hiện viết script đánh giá V0 \] Baseline** — LLM trả lời trực tiếp, không có gì thêm (để so sánh).:  1 LLM  
2. **\[Đạt\] Medical reasoning agent** — agent chuyên suy luận y khoa (chain-of-thought, phân tích triệu chứng...).: 1 LLM \+ prompt  
3. **\[Đạt\] RAG module** — truy xuất tài liệu y khoa liên quan (sách giáo khoa, guideline...) để hỗ trợ trả lời.: 1 LLM \+ RAG \+ (prompt) \-\> output  
4. **\[Hoàng\] Short-term memory** — bộ nhớ tạm trong 1 phiên làm việc (ví dụ nhớ các bước suy luận trước đó của câu hỏi hiện tại). 1 LLM \+ **Short-term memory** \+ (prompt) \-\> output  
5. **\[Hoàng\] Long-term memory** — bộ nhớ dài hạn, lưu lại kinh nghiệm/lỗi từ các câu hỏi trước để cải thiện sau này. 1 LLM \+ **Long-term memory** \+ (prompt) \-\> output  
6. **\[Chi\] Multi-agent workflow** — ít nhất 3 agent làm việc cùng nhau (ví dụ: agent suy luận, agent truy xuất, agent kiểm chứng/verifier). 3 LLM \-\> output  
7. **Evolutionary optimization** (tùy chọn) — dùng thuật toán tiến hóa để tối ưu prompt/tham số hệ thống.

Input \-\> Agent truy xuất \-\> RAG 

		|  
		V

	Agent suy luận

		|  
		V

	Agent verifier  \-\> Output  
		

Trao đổi:

- Nền tảng để kết nối agent: langchain  
- Rag module: nhiều loại langchain, landeDB? Framework Llama Index   
- Model LLM: Dùng Web \-\> API để tiết kiệm chi phí 

### **3\. Các phiên bản hệ thống cần so sánh (Ablation study)**

Đây là phần **rất quan trọng** — bạn cần chạy 4-5 biến thể để xem từng thành phần đóng góp bao nhiêu:

* **V0**: LLM baseline (không có gì thêm) v  
* **V1**: chỉ có RAG   
* **V2**: multi-agent nhưng không có memory  
* **V3**: hệ thống đầy đủ (full system)  
* **V4** (tùy chọn): full system nhưng bỏ verifier

→ Mục đích: chứng minh bằng số liệu rằng "thêm module X giúp tăng độ chính xác bao nhiêu".

### **4\. Quy trình benchmark (rất quan trọng để tránh overfitting)**

* **Dev set** (100-150 câu): chỉ dùng để debug, tinh chỉnh prompt/tham số — **không được** dùng để báo cáo kết quả cuối.  
* **Test set chính thức** (toàn bộ MedQA-USMLE, 1273 câu): chỉ chạy **1 lần duy nhất** để lấy kết quả cuối cùng, không được chạy đi chạy lại để "chọn kết quả đẹp" (tránh data leakage/overfitting).

### **5\. Các chỉ số đánh giá (metrics)**

* **Accuracy** \= số câu đúng / tổng số câu (với test set chính thức là chia cho 1273\)  
* **Invalid response rate**: tỷ lệ câu trả lời không hợp lệ (model không đưa ra được đáp án rõ ràng)  
* **Accuracy gain** \= Accuracy(V3) − Accuracy(V0) → đo mức cải thiện của hệ thống đầy đủ so với baseline  
* **Win/Loss/Tie**: so từng câu, hệ thống nào đúng/sai/hòa  
* **Bootstrap CI, McNemar test**: kiểm định thống kê xem sự khác biệt có ý nghĩa hay chỉ là ngẫu nhiên  
* **Cost & latency**: chi phí API và thời gian phản hồi

### **6\. Sản phẩm nộp (deliverables)**

* Code chạy được (baseline \+ full system)  
* File dự đoán (prediction files) cho từng biến thể  
* Script đánh giá  
* Report theo cấu trúc chuẩn (Intro → Kiến trúc → Thiết kế → Thực nghiệm → Kết quả → Phân tích lỗi → Discussion → Limitations → Conclusion)  
* Sơ đồ kiến trúc, slide thuyết trình 7-10 phút  
* Prompts đã dùng, cấu hình (model, temperature, seed...) để đảm bảo **có thể tái lập** (reproducibility)

### **7\. Lưu ý quan trọng**

Mục **18** cho biết: hệ thống bạn nộp ở giữa kỳ sẽ **được dùng lại làm mục tiêu tấn công/phòng thủ (attack/defense)** ở đồ án cuối kỳ — nên khi thiết kế, bạn nên nghĩ trước đến việc hệ thống có thể bị "tấn công" (ví dụ prompt injection, adversarial input) như thế nào.

