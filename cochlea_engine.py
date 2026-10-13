import json
import time
import os
from cells.hair_cell import HairCell

class CochleaEngine:
    def __init__(self):
        # 2 tế bào lông rung: 1 bắt âm trầm (Bass), 1 bắt âm bổng (Treble)
        self.cell_bass = HairCell("k_bass", filter_len=4)
        self.cell_treble = HairCell("k_treble", filter_len=4)

    # 1. CHIỀU XUÔI: NGHE SÓNG ÂM -> PHÁT XUNG THÍNH GIÁC CHO NÃO
    def listen_and_emit(self, wave_signal):
        e_bass = self.cell_bass.scan_wave(wave_signal)
        e_treble = self.cell_treble.scan_wave(wave_signal)
        
        # Chuẩn hóa năng lượng về thang đo [0.0, 1.0]
        features = [round(min(e_bass, 1.0), 2), round(min(e_treble, 1.0), 2)]
        timestamp = int(time.time())

        packet_forward = {
            "contract": "bodhi.audio.v1",
            "source": "cochlea_ear",
            "timestamp": timestamp,
            "features": features
        }

        with open("synapse_forward.json", "w") as f:
            json.dump(packet_forward, f, indent=4)

        print(f"👂 [ỐC TAI ĐÃ NGHE] Sóng âm -> Tín hiệu [Trầm, Bổng]: {features} (Timestamp: {timestamp})")
        return packet_forward

    # 2. CHIỀU NGƯỢC (SPLIT LEARNING): ĐÓN ĐẠO HÀM TỪ NÃO ĐỂ TỰ CHỈNH DÂY ĐÀN
    def absorb_brain_feedback(self, backward_file="synapse_backward.json"):
        if not os.path.exists(backward_file):
            print("🍃 [MÔI TRƯỜNG TĨNH] Chưa có xung đạo hàm phản hồi từ Não.")
            return

        try:
            with open(backward_file, "r") as f:
                data = json.load(f)
            grads = data.get("gradients_for_eye", [0.0, 0.0])
            print(f"⚡ [NHẬN ĐẠO HÀM TỪ NÃO] Gradient dL/df = {grads}")

            # 2 Sợi lông tự điều chỉnh độ căng của màng rung
            self.cell_bass.tune_from_brain_gradient(grads[0])
            self.cell_treble.tune_from_brain_gradient(grads[1])
            print("🔧 [TIẾN HÓA] Ốc tai đã tự chỉnh lại tần số rung của lông cảm ứng!")
        except Exception as e:
            print(f"⚠️ Lỗi đọc xung đạo hàm: {e}")

if __name__ == "__main__":
    ear = CochleaEngine()
    
    # BƯỚC 1: Đón đạo hàm phản hồi từ Não
    ear.absorb_brain_feedback()

    # BƯỚC 2: Nghe một đoạn sóng âm thực tế
    # Ví dụ: Sóng dao động nhanh (Tiếng còi báo động / Âm bổng tần số cao)
    song_am_am_bong = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.8, -0.8]
    ear.listen_and_emit(song_am_am_bong)
