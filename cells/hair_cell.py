import random
import json
import os

class HairCell:
    def __init__(self, cell_id, filter_len=4, storage_dir="synapses"):
        self.id = cell_id
        self.filter_len = filter_len
        self.storage_path = os.path.join(storage_dir, f"{self.id}.json")
        self.os_dir = storage_dir
        
        self.kernel = []
        self.last_wave_window = []
        self.last_resonance = 0.0
        
        # Tự nạp túi nhớ hoặc tự khởi sinh
        self.load_self()

    def load_self(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                if len(data.get("k", [])) == self.filter_len:
                    self.kernel = data["k"]
                    return
            except Exception:
                pass

        # TỰ KHỞI SINH: Tạo độ dài sợi lông ngẫu nhiên
        self.kernel = [random.uniform(-0.5, 0.5) for _ in range(self.filter_len)]
        self.save_self()

    # QUÉT SÓNG ÂM THANH 1D: Phép tích chập 1 chiều (1D Convolution)
    def scan_wave(self, wave):
        n_samples = len(wave)
        max_energy = 0.0
        best_window = []

        # Trượt cửa sổ sợi lông dọc theo dòng thời gian của sóng âm
        for i in range(n_samples - self.filter_len + 1):
            window = wave[i : i + self.filter_len]
            # Tích chập 1D: Nhân sóng âm với tần số sợi lông
            resonance = sum(window[j] * self.kernel[j] for j in range(self.filter_len))
            # ReLU: Giữ lại năng lượng cộng hưởng dương
            resonance = max(0.0, resonance)
            
            if resonance > max_energy:
                max_energy = resonance
                best_window = window

        self.last_wave_window = best_window if best_window else wave[:self.filter_len]
        self.last_resonance = max_energy
        return self.last_resonance

    # CHIỀU NGƯỢC (SPLIT LEARNING): Vặn lại sợi lông theo ý của Não
    def tune_from_brain_gradient(self, grad, lr=0.1):
        if self.last_resonance > 0 and self.last_wave_window:
            for j in range(self.filter_len):
                self.kernel[j] += lr * grad * self.last_wave_window[j]
            self.save_self()

    def save_self(self):
        os.makedirs(self.os_dir, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({"k": self.kernel}, f, indent=4)
