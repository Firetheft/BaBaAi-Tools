import os
import csv
import numpy as np
import logging
import re

try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(NODE_DIR, "meodai_colornames.csv")

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
        logging.warning(f"[MeodaiColors] Invalid hex code skipped: {hex_str}")
        return None
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

class MeodaiColorNames:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_names = []
        self.rgb_values = []
        self.kdtree = None
        self._load_data()

    def _load_data(self):
        if not os.path.exists(CSV_PATH):
            self.logger.error(f"[MeodaiColors] Error: 'meodai_colornames.csv' not found at {CSV_PATH}")
            self.logger.error("Please download 'colornames.csv' from the 'meodai/color-names' project,")
            self.logger.error("rename it to 'meodai_colornames.csv', and place it in the 'BaBaAi-Tools/nodes/' directory.")
            return

        if not SCIPY_AVAILABLE:
            self.logger.error("[MeodaiColors] 'scipy' library not found. Cannot build KD-Tree for color search.")
            self.logger.error("Please ensure 'scipy' is installed (it should be a dependency of 'scikit-learn').")
            return

        try:
            temp_rgb_values = []
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    next(reader)
                except StopIteration:
                    self.logger.error("[MeodaiColors] CSV file is empty.")
                    return
                    
                for row in reader:
                    if len(row) < 2:
                        continue
                    name = row[0]
                    hex_val = row[1]
                    rgb = hex_to_rgb(hex_val)
                    if rgb:
                        self.color_names.append(name)
                        temp_rgb_values.append(rgb)
            
            if not self.color_names:
                self.logger.error("[MeodaiColors] No valid color data loaded from CSV.")
                return

            self.rgb_values = np.array(temp_rgb_values)
            
            self.kdtree = KDTree(self.rgb_values)
            self.logger.info(f"[MeodaiColors] Successfully loaded {len(self.color_names)} colors and built KD-Tree.")
        
        except Exception as e:
            self.logger.error(f"[MeodaiColors] Error loading CSV or building KD-Tree: {e}")

    def get_closest_color_name(self, rgb_tuple):
        if self.kdtree is None:
            if not hasattr(self, '_warned_kdtree'):
                self.logger.warning("[MeodaiColors] KD-Tree not available. Returning 'Unknown'.")
                self._warned_kdtree = True
            return "Unknown"
        
        try:
            distance, index = self.kdtree.query(rgb_tuple)
            return self.color_names[index]
        except Exception as e:
            self.logger.error(f"[MeodaiColors] Error querying KD-Tree: {e}")
            return "Unknown"

MEODAI_COLORS = MeodaiColorNames()