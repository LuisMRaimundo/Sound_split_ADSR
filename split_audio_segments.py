"""
Optimized Audio Segment Splitter - Attack, Sustain, Decay, Release Detection

Features:
- Smart energy-guided detection (default) with proportional anchors
- Advanced derivative + spectral-flux mode
- Pitch-stable sustain refinement
- Zero-crossing cuts, cosine fades, manual review UI
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import soundfile as sf
import numpy as np
import threading
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
import multiprocessing as mp
import json
from datetime import datetime

import audio_segment_core as segcore
from audio_segment_core import SegmentConfig, ALL_PRESETS, PRESETS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class OptimizedAudioSplitter:
    """
    Optimized audio segment splitter with energy-based detection algorithms.
    """
    
    # Supported audio formats (librosa can handle most common formats)
    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.aif', '.aiff', '.ogg', '.m4a', '.wma', '.mp4', '.mka'}
    
    # Configurable parameters
    DEFAULT_FADE_MS = 50.0  # Increased default for better click prevention
    DEFAULT_TRIM_DB = 60.0
    DEFAULT_ATTACK_THRESHOLD = 0.9  # 90% of peak energy for attack
    DEFAULT_DECAY_THRESHOLD = 0.50  # 50% of peak for decay onset
    DEFAULT_SUSTAIN_VARIANCE_THRESHOLD = 0.2  # 20% variance for sustain plateau (relaxed)
    DEFAULT_MIN_SUSTAIN_DURATION = 1.0  # Minimum 1.0s for sustain (safe default)
    DEFAULT_MIN_SUSTAIN_FRAMES = 40  # Minimum frames in sustain for stability
    DEFAULT_PITCH_STABILITY_CENTS = 5.0  # Max pitch std (cents) for stationary sustain
    DEFAULT_PITCH_WINDOW_DURATION = 0.5  # Target window length for pitch stability (seconds)
    DEFAULT_ZERO_CROSSING_SEARCH_MS = 100.0  # Increased to 100ms for better zero crossing finding
    DEFAULT_FRAME_LENGTH = 1024
    DEFAULT_HOP_LENGTH = 512
    DEFAULT_FADE_TYPE = "cosine"  # Cosine fade for smoother transitions
    
    PRESETS = ALL_PRESETS

    def __init__(self, root):
        self.root = root
        self.root.title("Optimized Audio Segment Splitter v3.0 - Smart Detection")
        self.root.geometry("1000x850")
        
        # Configuration
        self.source_folder = tk.StringVar()
        self.fade_ms = tk.DoubleVar(value=self.DEFAULT_FADE_MS)
        self.attack_threshold = tk.DoubleVar(value=self.DEFAULT_ATTACK_THRESHOLD)
        self.decay_threshold = tk.DoubleVar(value=self.DEFAULT_DECAY_THRESHOLD)
        self.fade_type = tk.StringVar(value=self.DEFAULT_FADE_TYPE)
        self.use_smart_mode = tk.BooleanVar(value=True)  # Energy-guided (recommended)
        self.use_advanced_mode = tk.BooleanVar(value=False)  # Derivative + flux mode
        self.use_multiprocessing = tk.BooleanVar(value=False)  # Off by default (thread-safe sequential)
        self._state_lock = threading.Lock()
        self.max_workers = tk.IntVar(value=min(4, mp.cpu_count()))  # Number of parallel workers
        self.pitch_stability_cents = tk.DoubleVar(value=self.DEFAULT_PITCH_STABILITY_CENTS)
        self.pitch_window_duration = tk.DoubleVar(value=self.DEFAULT_PITCH_WINDOW_DURATION)
        self.pitch_refine_mode = tk.StringVar(value="expand")
        
        # Preset configuration
        self.preset_name = tk.StringVar(value="Medium (1.5-3.0s)")
        self.mean_sound_length = tk.DoubleVar(value=2.0)  # Mean sound length in seconds
        self.attack_pct = tk.DoubleVar(value=0.15)  # Attack percentage (15%)
        self.sustain_pct = tk.DoubleVar(value=0.60)  # Sustain percentage (60%)
        self.decay_pct = tk.DoubleVar(value=0.25)  # Decay percentage (25%)
        self.min_sustain_duration = tk.DoubleVar(value=0.35)  # Minimum sustain duration in seconds
        
        # State
        self.is_running = False
        self.segment_info: List[Dict] = []
        self.log_lines: List[str] = []
        
        # Review window components
        self.review_window = None
        self.review_tree = None
        self.review_fig = None
        self.review_ax = None
        self.review_canvas = None
        self.current_file_for_plot = None
        self.attack_line = None
        self.decay_line = None
        self.dragging_line = None
        self.active_line = None
        self.nudge_seconds = 0.005
        self.pending_apply_id = None
        self.pending_att = None
        self.pending_dec = None
        
        # Cache for loaded audio (to avoid reloading)
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        # Manual overrides to prevent auto re-detection from overwriting edits
        self.manual_overrides: Dict[str, Tuple[float, float]] = {}
        self._last_pitch_refine_info: Dict[str, Optional[float]] = {}
        self._last_trim: Optional[segcore.TrimInfo] = None

        self._build_ui()

    def _config_from_ui(self) -> SegmentConfig:
        """Build detection config from current UI values."""
        return SegmentConfig(
            trim_db=self.DEFAULT_TRIM_DB,
            attack_threshold=float(self.attack_threshold.get()),
            decay_threshold=float(self.decay_threshold.get()),
            attack_pct=float(self.attack_pct.get()),
            sustain_pct=float(self.sustain_pct.get()),
            decay_pct=float(self.decay_pct.get()),
            min_sustain_duration=float(self.min_sustain_duration.get()),
            pitch_window_duration=float(self.pitch_window_duration.get()),
            pitch_stability_cents=float(self.pitch_stability_cents.get()),
            pitch_refine_mode=str(self.pitch_refine_mode.get()),
            use_advanced=bool(self.use_advanced_mode.get()),
            use_smart=bool(self.use_smart_mode.get()) and not bool(self.use_advanced_mode.get()),
        )
    
    def _build_ui(self):
        """Build the user interface."""
        # Source folder selection
        frame_src = ttk.LabelFrame(self.root, text="Source Folder", padding=10)
        frame_src.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame_src, text="Folder:").pack(side=tk.LEFT)
        ttk.Entry(frame_src, textvariable=self.source_folder, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_src, text="Browse", command=self._browse_folder).pack(side=tk.LEFT)
        
        # Preset Configuration
        frame_preset = ttk.LabelFrame(self.root, text="Preset Configuration", padding=10)
        frame_preset.pack(fill=tk.X, padx=10, pady=5)
        
        preset_grid = ttk.Frame(frame_preset)
        preset_grid.pack(fill=tk.X)
        
        ttk.Label(preset_grid, text="Preset:").grid(row=0, column=0, padx=5, sticky="w")
        preset_combo = ttk.Combobox(preset_grid, textvariable=self.preset_name,
                                   values=list(self.PRESETS.keys()), state="readonly", width=20)
        preset_combo.grid(row=0, column=1, padx=5, sticky="w")
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)
        
        ttk.Label(preset_grid, text="Mean Sound Length (s):").grid(row=0, column=2, padx=5, sticky="w")
        mean_length_spin = ttk.Spinbox(preset_grid, from_=0.1, to=30.0, 
                                      textvariable=self.mean_sound_length, 
                                      increment=0.1, width=10, format="%.2f")
        mean_length_spin.grid(row=0, column=3, padx=5)
        mean_length_spin.bind("<FocusOut>", self._on_mean_length_changed)
        mean_length_spin.bind("<Return>", self._on_mean_length_changed)
        
        ttk.Button(preset_grid, text="Auto-Detect Mean Length", 
                  command=self._auto_detect_mean_length).grid(row=0, column=4, padx=5)
        
        ttk.Button(preset_grid, text="Apply Preset", 
                  command=self._apply_preset).grid(row=0, column=5, padx=5)
        
        # Advanced parameters
        frame_params = ttk.LabelFrame(self.root, text="Segmentation Parameters", padding=10)
        frame_params.pack(fill=tk.X, padx=10, pady=5)
        
        params_grid = ttk.Frame(frame_params)
        params_grid.pack(fill=tk.X)
        
        # Row 0: Attack and Sustain percentages
        ttk.Label(params_grid, text="Attack %:").grid(row=0, column=0, padx=5, sticky="w")
        attack_pct_spin = ttk.Spinbox(params_grid, from_=0.05, to=0.40, 
                                      textvariable=self.attack_pct, 
                                      increment=0.01, width=10, format="%.2f")
        attack_pct_spin.grid(row=0, column=1, padx=5)
        attack_pct_spin.bind("<FocusOut>", self._validate_percentages)
        attack_pct_spin.bind("<Return>", self._validate_percentages)
        
        ttk.Label(params_grid, text="Sustain %:").grid(row=0, column=2, padx=5, sticky="w")
        sustain_pct_spin = ttk.Spinbox(params_grid, from_=0.30, to=0.80, 
                                       textvariable=self.sustain_pct, 
                                       increment=0.01, width=10, format="%.2f")
        sustain_pct_spin.grid(row=0, column=3, padx=5)
        sustain_pct_spin.bind("<FocusOut>", self._validate_percentages)
        sustain_pct_spin.bind("<Return>", self._validate_percentages)
        
        ttk.Label(params_grid, text="Decay %:").grid(row=0, column=4, padx=5, sticky="w")
        decay_pct_spin = ttk.Spinbox(params_grid, from_=0.10, to=0.50, 
                                     textvariable=self.decay_pct, 
                                     increment=0.01, width=10, format="%.2f")
        decay_pct_spin.grid(row=0, column=5, padx=5)
        decay_pct_spin.bind("<FocusOut>", self._validate_percentages)
        decay_pct_spin.bind("<Return>", self._validate_percentages)
        
        # Row 1: Fade and thresholds
        ttk.Label(params_grid, text="Fade (ms):").grid(row=1, column=0, padx=5, sticky="w")
        ttk.Spinbox(params_grid, from_=5, to=200, textvariable=self.fade_ms, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(params_grid, text="Attack Threshold:").grid(row=1, column=2, padx=5, sticky="w")
        ttk.Spinbox(params_grid, from_=0.5, to=0.99, textvariable=self.attack_threshold, 
                   increment=0.05, width=10, format="%.2f").grid(row=1, column=3, padx=5)
        
        ttk.Label(params_grid, text="Decay Threshold:").grid(row=1, column=4, padx=5, sticky="w")
        ttk.Spinbox(params_grid, from_=0.1, to=0.9, textvariable=self.decay_threshold, 
                   increment=0.05, width=10, format="%.2f").grid(row=1, column=5, padx=5)
        
        # Row 2: Additional parameters
        ttk.Label(params_grid, text="Min Sustain (s):").grid(row=2, column=0, padx=5, sticky="w")
        ttk.Spinbox(params_grid, from_=0.05, to=2.0, textvariable=self.min_sustain_duration, 
                   increment=0.05, width=10, format="%.2f").grid(row=2, column=1, padx=5)
        
        ttk.Label(params_grid, text="Fade Type:").grid(row=2, column=2, padx=5, sticky="w")
        fade_combo = ttk.Combobox(params_grid, textvariable=self.fade_type, 
                                 values=["cosine", "hann", "linear"], state="readonly", width=10)
        fade_combo.grid(row=2, column=3, padx=5)
        
        ttk.Label(params_grid, text="Smart Mode:").grid(row=2, column=4, padx=5, sticky="w")
        ttk.Checkbutton(params_grid, variable=self.use_smart_mode,
                       text="Energy-guided (recommended)").grid(row=2, column=5, padx=5, sticky="w")

        ttk.Label(params_grid, text="Advanced Mode:").grid(row=3, column=0, padx=5, sticky="w")
        ttk.Checkbutton(params_grid, variable=self.use_advanced_mode,
                       text="Derivative + spectral flux").grid(row=3, column=1, columnspan=2, padx=5, sticky="w")

        ttk.Label(params_grid, text="Pitch Stability (¢ std):").grid(row=3, column=3, padx=5, sticky="w")
        ttk.Spinbox(params_grid, from_=1.0, to=20.0, textvariable=self.pitch_stability_cents,
                   increment=0.5, width=10, format="%.1f").grid(row=3, column=4, padx=5)

        ttk.Label(params_grid, text="Pitch Window (s):").grid(row=3, column=5, padx=5, sticky="w")
        ttk.Spinbox(params_grid, from_=0.1, to=2.0, textvariable=self.pitch_window_duration,
                   increment=0.05, width=10, format="%.2f").grid(row=4, column=0, padx=5)

        ttk.Label(params_grid, text="Pitch Refine:").grid(row=4, column=1, padx=5, sticky="w")
        ttk.Combobox(
            params_grid,
            textvariable=self.pitch_refine_mode,
            values=["expand", "annotate", "crop"],
            state="readonly",
            width=12,
        ).grid(row=4, column=2, padx=5, sticky="w")

        ttk.Label(params_grid, text="Parallel batch:").grid(row=4, column=3, padx=5, sticky="w")
        ttk.Checkbutton(params_grid, variable=self.use_multiprocessing,
                       text="Thread pool (parallel batch)").grid(row=4, column=4, padx=5, sticky="w")

        ttk.Label(params_grid, text="Workers:").grid(row=4, column=5, padx=5, sticky="w")
        ttk.Spinbox(params_grid, from_=1, to=mp.cpu_count(), textvariable=self.max_workers,
                   width=10).grid(row=5, column=0, padx=5)
        
        # Log area
        frame_log = ttk.LabelFrame(self.root, text="Log", padding=10)
        frame_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.txt_log = tk.Text(frame_log, state=tk.DISABLED, font=("Consolas", 9), bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(frame_log, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=scrollbar.set)
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Run and Clear buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self.btn_clear = ttk.Button(btn_frame, text="Clear", command=self._clear)
        self.btn_clear.pack(side=tk.LEFT, padx=(0, 10), ipady=5)
        self.btn_run = ttk.Button(btn_frame, text="► RUN OPTIMIZED SPLIT", command=self._run_batch)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
    
    def _clear(self):
        """Clear all state for a fresh run on a new collection."""
        if self.is_running:
            return
        self.segment_info = []
        self.log_lines = []
        self._audio_cache.clear()
        self.manual_overrides.clear()
        self._last_pitch_refine_info = {}
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.config(state=tk.DISABLED)
        self.progress.configure(value=0)
        if self.review_window is not None:
            self._close_review_window()
        self._log("Cleared. Select a folder and run again.")
    
    def _log(self, msg: str):
        """Thread-safe logging."""
        self.log_lines.append(msg)
        self.root.after(0, lambda: self._update_log(msg))
    
    def _update_log(self, msg: str):
        """Update log display (must be called from main thread)."""
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)
    
    def _browse_folder(self):
        """Browse for source folder."""
        d = filedialog.askdirectory()
        if d:
            self.source_folder.set(d)
    
    def _on_preset_changed(self, event=None):
        """Handle preset selection change."""
        preset_name = self.preset_name.get()
        if preset_name in self.PRESETS:
            preset = self.PRESETS[preset_name]
            # Update mean length based on preset if not custom
            if preset_name != "Custom":
                # Suggest mean length based on preset
                if preset_name == "Very Short (< 0.5s)":
                    self.mean_sound_length.set(0.3)
                elif preset_name == "Short (0.5-1.5s)":
                    self.mean_sound_length.set(1.0)
                elif preset_name == "Medium (1.5-3.0s)":
                    self.mean_sound_length.set(2.0)
                elif preset_name == "Long (3.0-6.0s)":
                    self.mean_sound_length.set(4.0)
                elif preset_name == "Very Long (> 6.0s)":
                    self.mean_sound_length.set(8.0)
            self._apply_preset()
    
    def _on_mean_length_changed(self, event=None):
        """Handle mean length change - auto-select appropriate preset."""
        mean_len = self.mean_sound_length.get()
        if mean_len < 0.5:
            suggested_preset = "Very Short (< 0.5s)"
        elif mean_len < 1.5:
            suggested_preset = "Short (0.5-1.5s)"
        elif mean_len < 3.0:
            suggested_preset = "Medium (1.5-3.0s)"
        elif mean_len < 6.0:
            suggested_preset = "Long (3.0-6.0s)"
        else:
            suggested_preset = "Very Long (> 6.0s)"
        
        # Only auto-select if current preset is not Custom
        if self.preset_name.get() != "Custom":
            self.preset_name.set(suggested_preset)
            self._apply_preset()
    
    def _apply_preset(self):
        """Apply preset configuration to parameters."""
        preset_name = self.preset_name.get()
        if preset_name in self.PRESETS:
            preset = self.PRESETS[preset_name]
            self.attack_pct.set(preset["attack_pct"])
            self.sustain_pct.set(preset["sustain_pct"])
            self.decay_pct.set(preset["decay_pct"])
            self.fade_ms.set(preset["fade_ms"])
            self.min_sustain_duration.set(preset["min_sustain_duration"])
            self.attack_threshold.set(preset["attack_threshold"])
            self.decay_threshold.set(preset["decay_threshold"])
            if "use_advanced" in preset:
                self.use_advanced_mode.set(bool(preset["use_advanced"]))
            if "use_smart" in preset:
                self.use_smart_mode.set(bool(preset["use_smart"]))
            if "pitch_stability_cents" in preset:
                self.pitch_stability_cents.set(float(preset["pitch_stability_cents"]))
            if "pitch_refine_mode" in preset:
                self.pitch_refine_mode.set(str(preset["pitch_refine_mode"]))
            self._validate_percentages()
            self._log(f"Applied preset: {preset_name}")
    
    def _validate_percentages(self, event=None):
        """Validate that attack + sustain + decay percentages sum to approximately 1.0."""
        attack = float(self.attack_pct.get())
        sustain = float(self.sustain_pct.get())
        decay = float(self.decay_pct.get())

        # Guard against invalid values outside expected range
        if any(v < 0.0 or v > 1.0 for v in (attack, sustain, decay)):
            self.attack_pct.set(0.15)
            self.sustain_pct.set(0.60)
            self.decay_pct.set(0.25)
            self._log("Reset percentages to defaults (out of range)")
            return
        total = attack + sustain + decay
        
        if abs(total - 1.0) > 0.01:  # Allow 1% tolerance
            # Normalize to sum to 1.0
            if total > 0:
                self.attack_pct.set(attack / total)
                self.sustain_pct.set(sustain / total)
                self.decay_pct.set(decay / total)
                self._log(f"Normalized percentages to sum to 1.0 (was {total:.3f})")
            else:
                # Fallback to defaults
                self.attack_pct.set(0.15)
                self.sustain_pct.set(0.60)
                self.decay_pct.set(0.25)
                self._log("Reset percentages to defaults (sum was 0)")
    
    def _auto_detect_mean_length(self):
        """Auto-detect mean sound length from files in source folder."""
        folder = self.source_folder.get()
        if not folder or not Path(folder).exists():
            messagebox.showwarning("No Folder", "Please select a source folder first.")
            return
        
        try:
            # Use all supported audio formats for auto-detection
            audio_files = []
            for ext in self.SUPPORTED_AUDIO_FORMATS:
                audio_files.extend(Path(folder).glob(f"*{ext}"))
                audio_files.extend(Path(folder).glob(f"*{ext.upper()}"))  # Also check uppercase extensions
            audio_files = [f for f in audio_files if not f.stem.lower().endswith("_backup")]
            
            if not audio_files:
                format_list = ', '.join(sorted(self.SUPPORTED_AUDIO_FORMATS))
                messagebox.showwarning("No Audio Files", f"No audio files found in the selected folder.\n\nSupported formats: {format_list}")
                return
            
            self._log(f"Analyzing {len(audio_files)} audio files to detect mean length...")
            lengths = []
            
            for audio_file in audio_files[:100]:  # Limit to 100 files for speed
                try:
                    y, sr = librosa.load(str(audio_file), sr=None, duration=None)
                    # Trim to get active length
                    y_trimmed, index = librosa.effects.trim(y, top_db=self.DEFAULT_TRIM_DB)
                    active_len = (index[1] - index[0]) / sr
                    if active_len > 0.01:  # Ignore very short or invalid
                        lengths.append(active_len)
                except Exception as e:
                    logger.warning(f"Could not analyze {audio_file.name}: {e}")
                    continue
            
            if lengths:
                mean_length = np.mean(lengths)
                self.mean_sound_length.set(round(mean_length, 2))
                self._log(f"Detected mean sound length: {mean_length:.2f}s (from {len(lengths)} files)")
                self._on_mean_length_changed()  # Auto-select appropriate preset
                messagebox.showinfo("Auto-Detection Complete", 
                                  f"Mean sound length: {mean_length:.2f}s\n"
                                  f"Based on {len(lengths)} files\n"
                                  f"Preset auto-selected: {self.preset_name.get()}")
            else:
                messagebox.showwarning("No Valid Files", "Could not determine mean length from files.")
        except Exception as e:
            logger.error(f"Error in auto-detect: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to auto-detect mean length:\n{e}")
    
    def detect_segments(self, y: np.ndarray, sr: int, file_path: Optional[Path] = None) -> Tuple[float, float, float]:
        """Detect segment boundaries (absolute file times)."""
        result = segcore.detect_segments(y, sr, self._config_from_ui(), file_path)
        self._last_trim = result.trim
        self._last_pitch_refine_info = result.pitch_refine
        return result.t_att, result.t_dec, result.t_end

    def _validate_segments(self, t_att: float, t_dec: float, t_end: float,
                          min_duration: float = 0.01) -> bool:
        return segcore.validate_segments(t_att, t_dec, t_end, min_duration)

    def _extract_and_fade_segments(self, y: np.ndarray, sr: int,
                                   t_att: float, t_dec: float, t_end: float) -> Tuple[Dict[str, np.ndarray], int, int, int]:
        trim = self._last_trim
        if trim is None:
            _, trim = segcore.trim_active_region(y, sr, self.DEFAULT_TRIM_DB)
        return segcore.extract_and_fade_segments(
            y, sr, t_att, t_dec, t_end, trim,
            float(self.fade_ms.get()), str(self.fade_type.get()),
        )

    def _soundfile_format_for_extension(self, ext: str) -> Optional[str]:
        """Return a soundfile format string for a given extension."""
        format_map = {
            '.wav': 'WAV',
            '.aif': 'AIFF',
            '.aiff': 'AIFF',
            '.flac': 'FLAC',
            '.ogg': 'OGG'
        }
        return format_map.get(ext.lower())

    def _write_audio(self, output_path: Path, audio: np.ndarray, sr: int):
        """Write audio with explicit format for extensions like .aif."""
        sf_format = self._soundfile_format_for_extension(output_path.suffix)
        if sf_format:
            sf.write(output_path, audio, sr, format=sf_format)
        else:
            sf.write(output_path, audio, sr)
    
    def process_file(self, f_path: Path, out_dir: Path) -> str:
        """
        Process a single audio file and split into segments.
        
        Args:
            f_path: Path to audio file
            out_dir: Output directory
        
        Returns:
            Status message
        """
        try:
            # Load audio (with caching for review)
            if str(f_path) not in self._audio_cache:
                y, sr = librosa.load(f_path, sr=None)
                self._audio_cache[str(f_path)] = (y, sr)
            else:
                y, sr = self._audio_cache[str(f_path)]
            
            # Detect segments unless manual override exists
            file_key = str(f_path)
            info = next((d for d in self.segment_info if d["file_path"] == file_key), None)
            if file_key in self.manual_overrides:
                t_att, t_dec = self.manual_overrides[file_key]
                if info and "t_end" in info:
                    t_end = info["t_end"]
                else:
                    _, _, t_end = self.detect_segments(y, sr, file_path=f_path)
                if self._last_trim is None:
                    _, self._last_trim = segcore.trim_active_region(y, sr, self.DEFAULT_TRIM_DB)
                pitch_info = {
                    "used": False,
                    "std_cents": None,
                    "window_start": None,
                    "window_end": None,
                    "window_duration": None,
                    "expected_note_hz": None,
                    "mean_abs_cents_from_note": None
                }
            else:
                t_att, t_dec, t_end = self.detect_segments(y, sr, file_path=f_path)
                pitch_info = dict(self._last_pitch_refine_info) if self._last_pitch_refine_info else {
                    "used": False,
                    "std_cents": None,
                    "window_start": None,
                    "window_end": None,
                    "window_duration": None
                }
            
            # Validate
            if not self._validate_segments(t_att, t_dec, t_end):
                return f"Error: Invalid segment boundaries detected"
            
            # Extract segments and apply fades (shared logic)
            parts, idx_att, idx_dec, idx_end = self._extract_and_fade_segments(y, sr, t_att, t_dec, t_end)
            
            # Write segments
            for folder, audio in parts.items():
                target_dir = out_dir / folder
                target_dir.mkdir(exist_ok=True, parents=True)
                if len(audio) > 0:
                    if folder == "_Full_Active_Sound":
                        tag = "FullActive"
                    elif folder == "_Release_Silence":
                        tag = "Release"
                    else:
                        tag = folder.strip('_')
                    output_path = target_dir / f"{f_path.stem}_{tag}{f_path.suffix}"
                    self._write_audio(output_path, audio, sr)
            
            trim = self._last_trim
            idx_start = trim.idx_start if trim else 0
            info = {
                "file_path": str(f_path),
                "sr": sr,
                "t_start": idx_start / sr,
                "t_att": idx_att / sr,
                "t_dec": idx_dec / sr,
                "t_end": idx_end / sr,
                "dur_att": (idx_att - idx_start) / sr,
                "dur_sus": (idx_dec - idx_att) / sr,
                "dur_dec": (idx_end - idx_dec) / sr,
                "dur_rel": (len(y) - idx_end) / sr,
                "pitch_refine": pitch_info,
            }
            with self._state_lock:
                self.segment_info = [i for i in self.segment_info if i["file_path"] != str(f_path)] + [info]
            
            if pitch_info.get("used"):
                msg = (
                    f"Att: {info['dur_att']:.2f}s | Sus: {info['dur_sus']:.2f}s | Dec: {info['dur_dec']:.2f}s"
                    f" | PitchWin: {pitch_info['window_start']:.2f}-{pitch_info['window_end']:.2f}s"
                    f" (σ={pitch_info['std_cents']:.2f}¢)"
                )
                if pitch_info.get("mean_abs_cents_from_note") is not None:
                    msg += f" | Δnote={pitch_info['mean_abs_cents_from_note']:.2f}¢"
                return msg
            return f"Att: {info['dur_att']:.2f}s | Sus: {info['dur_sus']:.2f}s | Dec: {info['dur_dec']:.2f}s"
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            if "NoBackendError" in error_type or "backend" in error_msg.lower():
                return f"Error: Audio backend not available. Install ffmpeg or soundfile."
            elif "file" in error_msg.lower() and "not found" in error_msg.lower():
                return f"Error: File not found"
            elif "format" in error_msg.lower() or "codec" in error_msg.lower():
                return f"Error: Unsupported audio format"
            else:
                logger.error(f"Error processing {f_path.name}: {e}", exc_info=True)
                return f"Error: {error_type}: {error_msg}"
    
    def _apply_manual_segment(self, file_path: str, t_att_new: float, t_dec_new: float):
        """Apply manually adjusted segment boundaries."""
        try:
            f_path = Path(file_path)
            
            # Load from cache or file
            if file_path not in self._audio_cache:
                y, sr = librosa.load(f_path, sr=None)
                self._audio_cache[file_path] = (y, sr)
            else:
                y, sr = self._audio_cache[file_path]
            
            info = next((d for d in self.segment_info if d["file_path"] == file_path), None)
            if not info:
                messagebox.showerror("Error", "File info not found")
                return
            
            t_end = info["t_end"]
            
            # Validate new boundaries
            if not self._validate_segments(t_att_new, t_dec_new, t_end):
                messagebox.showerror("Error", "Invalid segment boundaries")
                return
            
            # Skip duplicate applies (prevents repeated logs)
            last = self.manual_overrides.get(file_path)
            if last is not None:
                if abs(last[0] - t_att_new) < 1e-4 and abs(last[1] - t_dec_new) < 1e-4:
                    return
            
            # Persist manual override to prevent auto re-detection overwrites
            self.manual_overrides[file_path] = (t_att_new, t_dec_new)
            # Cancel any pending apply to avoid stale updates
            if self.pending_apply_id:
                try:
                    self.root.after_cancel(self.pending_apply_id)
                except Exception:
                    pass
                self.pending_apply_id = None
                self.pending_att = None
                self.pending_dec = None
            
            # Extract segments and apply fades (shared logic)
            parts, idx_att, idx_dec, idx_end = self._extract_and_fade_segments(y, sr, t_att_new, t_dec_new, t_end)
            
            # Write segments
            out_dir = Path(self.source_folder.get())
            for folder, audio in parts.items():
                target_dir = out_dir / folder
                target_dir.mkdir(exist_ok=True, parents=True)
                if len(audio) > 0:
                    tag = folder.strip('_') if folder != "_Release_Silence" else "Release"
                    output_path = target_dir / f"{f_path.stem}_{tag}{f_path.suffix}"
                    self._write_audio(output_path, audio, sr)
            
            # Update info
            trim = self._last_trim
            idx_start = trim.idx_start if trim else 0
            info.update({
                "t_att": idx_att / sr,
                "t_dec": idx_dec / sr,
                "dur_att": (idx_att - idx_start) / sr,
                "dur_sus": (idx_dec - idx_att) / sr,
                "dur_dec": (idx_end - idx_dec) / sr,
                "dur_rel": (len(y) - idx_end) / sr,
            })
            
            # Update UI
            if self.review_tree:
                self._populate_review_tree()
            if self.review_ax:
                self._plot_segmentation(file_path)
            
            self._log(f"[MANUAL] {f_path.name} updated with fades.")
            
        except Exception as e:
            logger.error(f"Error in manual segment: {e}", exc_info=True)
            messagebox.showerror("Error", str(e))
    
    def _on_plot_press(self, event):
        """Handle mouse press on plot."""
        if event.inaxes != self.review_ax or event.xdata is None:
            return
        
        tol = 0.05
        self.dragging_line = None
        
        if self.attack_line and abs(event.xdata - float(self.attack_line.get_xdata()[0])) < tol:
            self.dragging_line = "attack"
            self.active_line = "attack"
        elif self.decay_line and abs(event.xdata - float(self.decay_line.get_xdata()[0])) < tol:
            self.dragging_line = "decay"
            self.active_line = "decay"
        
        if self.review_canvas:
            self.review_canvas.get_tk_widget().focus_set()
    
    def _on_plot_move(self, event):
        """Handle mouse move on plot."""
        if self.dragging_line is None or event.inaxes != self.review_ax or event.xdata is None:
            return
        
        x = max(0.0, event.xdata)
        if self.dragging_line == "attack":
            self.attack_line.set_xdata([x, x])
        else:
            self.decay_line.set_xdata([x, x])
        self.review_canvas.draw_idle()
    
    def _on_plot_release(self, event):
        """Handle mouse release on plot."""
        if self.dragging_line is None:
            return
        
        new_att = float(self.attack_line.get_xdata()[0])
        new_dec = float(self.decay_line.get_xdata()[0])
        self.dragging_line = None
        
        # Ensure decay is after attack
        new_dec = max(new_att + 0.05, new_dec)
        self.decay_line.set_xdata([new_dec, new_dec])
        self.review_canvas.draw_idle()
        
        self._apply_manual_segment(self.current_file_for_plot, new_att, new_dec)

    def _on_key_nudge(self, event):
        """Nudge attack/decay lines with arrow keys after selecting a line."""
        if not self.active_line or not self.current_file_for_plot:
            return
        if not self.attack_line or not self.decay_line:
            return

        info = next((d for d in self.segment_info if d["file_path"] == self.current_file_for_plot), None)
        if not info:
            return

        delta = self.nudge_seconds
        if event.state & 0x0001:  # Shift
            delta *= 5

        if event.keysym in ("Right", "Up"):
            step = delta
        elif event.keysym in ("Left", "Down"):
            step = -delta
        else:
            return

        att = float(self.attack_line.get_xdata()[0])
        dec = float(self.decay_line.get_xdata()[0])
        min_gap = 0.05
        t_end = info["t_end"]

        if self.active_line == "attack":
            att = max(0.0, min(att + step, dec - min_gap))
            self.attack_line.set_xdata([att, att])
        else:
            dec = max(att + min_gap, min(dec + step, t_end - 0.01))
            self.decay_line.set_xdata([dec, dec])

        self.review_canvas.draw_idle()
        self._schedule_manual_apply(att, dec)

    def _schedule_manual_apply(self, att: float, dec: float):
        """Debounce manual apply so key nudges stay responsive."""
        self.pending_att = att
        self.pending_dec = dec
        if self.pending_apply_id:
            try:
                self.root.after_cancel(self.pending_apply_id)
            except Exception:
                pass
        self.pending_apply_id = self.root.after(250, self._flush_manual_apply)

    def _flush_manual_apply(self):
        """Apply the last pending manual adjustment."""
        self.pending_apply_id = None
        if self.pending_att is None or self.pending_dec is None:
            return
        self._apply_manual_segment(self.current_file_for_plot, self.pending_att, self.pending_dec)
    
    def _open_review_window(self):
        """Open review window with segmentation results."""
        if not self.segment_info:
            messagebox.showinfo("Info", "No files processed yet.")
            return
        
        if self.review_window is not None:
            self.review_window.lift()
            return
        
        self.review_window = tk.Toplevel(self.root)
        self.review_window.title("Review Segmentation")
        self.review_window.geometry("1000x600")
        
        main = ttk.Frame(self.review_window)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: File list
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH)
        
        cols = ("File", "Att (s)", "Sus (s)", "Dec (s)", "Rel (s)")
        self.review_tree = ttk.Treeview(left, columns=cols, show="headings", height=20)
        for c in cols:
            self.review_tree.heading(c, text=c)
            self.review_tree.column(c, width=100)
        
        scrollbar_tree = ttk.Scrollbar(left, orient="vertical", command=self.review_tree.yview)
        self.review_tree.configure(yscrollcommand=scrollbar_tree.set)
        self.review_tree.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.review_tree.bind("<<TreeviewSelect>>", self._on_review_select)
        
        # Right: Waveform plot
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.review_fig, self.review_ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.review_canvas = FigureCanvasTkAgg(self.review_fig, master=right)
        canvas_widget = self.review_canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        self.review_fig.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.review_fig.canvas.mpl_connect("motion_notify_event", self._on_plot_move)
        self.review_fig.canvas.mpl_connect("button_release_event", self._on_plot_release)
        
        self._populate_review_tree()
        
        # Close handler
        self.review_window.protocol("WM_DELETE_WINDOW", self._close_review_window)
        self.review_window.bind("<KeyPress-Left>", self._on_key_nudge)
        self.review_window.bind("<KeyPress-Right>", self._on_key_nudge)
        self.review_window.bind("<KeyPress-Up>", self._on_key_nudge)
        self.review_window.bind("<KeyPress-Down>", self._on_key_nudge)
        canvas_widget.bind("<KeyPress-Left>", self._on_key_nudge)
        canvas_widget.bind("<KeyPress-Right>", self._on_key_nudge)
        canvas_widget.bind("<KeyPress-Up>", self._on_key_nudge)
        canvas_widget.bind("<KeyPress-Down>", self._on_key_nudge)
        canvas_widget.focus_set()
    
    def _close_review_window(self):
        """Close review window and clean up."""
        if self.review_window:
            plt.close(self.review_fig)
            self.review_window.destroy()
            self.review_window = None
            self.review_tree = None
            self.review_fig = None
            self.review_ax = None
            self.review_canvas = None
            self.active_line = None
            self.pending_apply_id = None
            self.pending_att = None
            self.pending_dec = None
    
    def _populate_review_tree(self):
        """Populate review tree with segment info."""
        if not self.review_tree:
            return
        
        for child in self.review_tree.get_children():
            self.review_tree.delete(child)
        
        for info in self.segment_info:
            self.review_tree.insert("", "end", iid=info["file_path"], values=(
                Path(info["file_path"]).name,
                f"{info['dur_att']:.2f}",
                f"{info['dur_sus']:.2f}",
                f"{info['dur_dec']:.2f}",
                f"{info['dur_rel']:.2f}"
            ))
    
    def _on_review_select(self, event):
        """Handle file selection in review tree."""
        sel = self.review_tree.selection()
        if sel:
            self._plot_segmentation(sel[0])
    
    def _plot_segmentation(self, file_path: str):
        """Plot waveform with segment boundaries."""
        if not self.review_ax:
            return
        
        info = next((d for d in self.segment_info if d["file_path"] == file_path), None)
        if not info:
            return
        
        # Load audio (from cache if available)
        if file_path not in self._audio_cache:
            y, sr = librosa.load(file_path, sr=None)
            self._audio_cache[file_path] = (y, sr)
        else:
            y, sr = self._audio_cache[file_path]
        
        # Downsample for plotting (if too long)
        max_samples = 50000
        if len(y) > max_samples:
            step = len(y) // max_samples
            y_plot = y[::step]
            t = np.linspace(0, len(y) / sr, len(y_plot))
        else:
            t = np.linspace(0, len(y) / sr, len(y))
            y_plot = y
        
        self.current_file_for_plot = file_path
        self.review_ax.clear()
        self.review_ax.plot(t, y_plot, linewidth=0.7, color='gray', alpha=0.7)
        
        # Draw boundaries
        self.attack_line = self.review_ax.axvline(info["t_att"], color="green", 
                                                  linestyle="--", linewidth=2, label="Attack")
        self.decay_line = self.review_ax.axvline(info["t_dec"], color="orange", 
                                                linestyle="--", linewidth=2, label="Decay")
        
        self.review_ax.axvline(info["t_end"], color="red", linestyle=":", 
                              linewidth=1, alpha=0.5, label="End")
        
        self.review_ax.set_title(Path(file_path).name, fontsize=10)
        self.review_ax.set_xlabel("Time (s)")
        self.review_ax.set_ylabel("Amplitude")
        self.review_ax.legend(loc='upper right', fontsize=8)
        self.review_ax.grid(True, alpha=0.3)
        
        self.review_canvas.draw()
    
    def _run_batch(self):
        """Start batch processing."""
        f = self.source_folder.get()
        if not f:
            messagebox.showwarning("Warning", "Please select a source folder.")
            return
        
        if not Path(f).exists():
            messagebox.showerror("Error", "Source folder does not exist.")
            return
        
        self.is_running = True
        self.log_lines = []
        self.progress.configure(value=0)
        self.btn_run.config(state=tk.DISABLED)
        
        threading.Thread(target=self._worker, args=(Path(f),), daemon=True).start()
    
    def _worker(self, folder: Path):
        """Worker thread for batch processing with optional multiprocessing."""
        try:
            files = [f for f in folder.glob("*") 
                    if f.suffix.lower() in self.SUPPORTED_AUDIO_FORMATS]
            files = [f for f in files if not f.stem.lower().endswith("_backup")]
            
            if not files:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No audio files found."))
                self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
                return
            
            total = len(files)
            use_mp = self.use_multiprocessing.get() if hasattr(self, 'use_multiprocessing') else False
            max_workers = self.max_workers.get() if hasattr(self, 'max_workers') else 1
            
            if use_mp and total > 1 and max_workers > 1:
                # Use multiprocessing
                self._log(f"Processing {total} file(s) with {max_workers} worker(s)...")
                self._worker_multiprocessing(files, folder, max_workers)
            else:
                # Use sequential processing
                self._log(f"Processing {total} file(s)...")
                self._worker_sequential(files, folder)
            
            # Export metadata if requested
            self._export_metadata(folder)
            
            self._log("Processing complete!")
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self.root.after(0, self._open_review_window)
            
        except Exception as e:
            logger.error(f"Error in worker: {e}", exc_info=True)
            self._log(f"Error: {e}")
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.is_running = False
    
    def _worker_sequential(self, files: List[Path], folder: Path):
        """Sequential processing of files."""
        total = len(files)
        for i, f in enumerate(files):
            if not self.is_running:
                break
            
            msg = self.process_file(f, folder)
            self._log(f"[{i+1}/{total}] {f.name} -> {msg}")
            
            progress = ((i + 1) / total) * 100
            self.root.after(0, lambda v=progress: self.progress.configure(value=v))
    
    def _worker_multiprocessing(self, files: List[Path], folder: Path, max_workers: int):
        """Multiprocessing batch processing."""
        # Note: Due to tkinter limitations, we process in chunks and update UI
        # Full multiprocessing would require pickling the entire class, which is complex
        # Instead, we use a hybrid approach: process files in parallel batches
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_file_wrapper(f_path: Path) -> Tuple[Path, str]:
            """Wrapper for process_file that can be used in threads."""
            try:
                msg = self.process_file(f_path, folder)
                return f_path, msg
            except Exception as e:
                logger.error(f"Error processing {f_path.name}: {e}")
                return f_path, f"Error: {e}"
        
        total = len(files)
        completed = 0
        
        # Use ThreadPoolExecutor (threads share memory, easier than multiprocessing)
        # For true multiprocessing, would need to refactor to avoid tkinter dependencies
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_file_wrapper, f): f for f in files}
            
            for future in as_completed(future_to_file):
                if not self.is_running:
                    break
                
                try:
                    f_path, msg = future.result()
                    completed += 1
                    self._log(f"[{completed}/{total}] {f_path.name} -> {msg}")
                    
                    progress = (completed / total) * 100
                    self.root.after(0, lambda v=progress: self.progress.configure(value=v))
                except Exception as e:
                    logger.error(f"Error in future: {e}")
                    completed += 1
    
    def _export_metadata(self, folder: Path):
        """Export segmentation metadata to JSON and CSV."""
        if not self.segment_info:
            return
        
        try:
            # Export to JSON
            json_path = folder / "segmentation_metadata.json"
            metadata = {
                "export_date": datetime.now().isoformat(),
                "parameters": {
                    "fade_ms": self.fade_ms.get(),
                    "attack_threshold": self.attack_threshold.get(),
                    "decay_threshold": self.decay_threshold.get(),
                    "fade_type": self.fade_type.get(),
                    "smart_mode": self.use_smart_mode.get() if hasattr(self, 'use_smart_mode') else True,
                    "advanced_mode": self.use_advanced_mode.get() if hasattr(self, 'use_advanced_mode') else False,
                    "multiprocessing": self.use_multiprocessing.get() if hasattr(self, 'use_multiprocessing') else False,
                    "pitch_stability_cents": self.pitch_stability_cents.get() if hasattr(self, 'pitch_stability_cents') else self.DEFAULT_PITCH_STABILITY_CENTS,
                    "pitch_window_duration": self.pitch_window_duration.get() if hasattr(self, 'pitch_window_duration') else self.DEFAULT_PITCH_WINDOW_DURATION,
                },
                "files": []
            }
            
            for info in self.segment_info:
                file_metadata = {
                    "file_path": str(info["file_path"]),
                    "sample_rate": info["sr"],
                    "segments": {
                        "attack_end": info["t_att"],
                        "decay_start": info["t_dec"],
                        "end": info["t_end"],
                        "durations": {
                            "attack": info["dur_att"],
                            "sustain": info["dur_sus"],
                            "decay": info["dur_dec"],
                            "release": info["dur_rel"]
                        },
                        "pitch_stability": info.get("pitch_refine", {})
                    }
                }
                metadata["files"].append(file_metadata)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Export to CSV
            csv_path = folder / "segmentation_metadata.csv"
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "File", "Sample Rate", "Attack End (s)", "Decay Start (s)", "End (s)",
                    "Attack Duration (s)", "Sustain Duration (s)", "Decay Duration (s)", "Release Duration (s)",
                    "Pitch Stable Used", "Pitch Std (cents)", "Pitch Window Start (s)", "Pitch Window End (s)", "Pitch Window Duration (s)",
                    "Expected Note (Hz)", "Mean Abs Cents From Note"
                ])
                for info in self.segment_info:
                    pitch = info.get("pitch_refine", {}) or {}
                    writer.writerow([
                        Path(info["file_path"]).name,
                        info["sr"],
                        f"{info['t_att']:.4f}",
                        f"{info['t_dec']:.4f}",
                        f"{info['t_end']:.4f}",
                        f"{info['dur_att']:.4f}",
                        f"{info['dur_sus']:.4f}",
                        f"{info['dur_dec']:.4f}",
                        f"{info['dur_rel']:.4f}",
                        pitch.get("used", False),
                        "" if pitch.get("std_cents") is None else f"{pitch['std_cents']:.4f}",
                        "" if pitch.get("window_start") is None else f"{pitch['window_start']:.4f}",
                        "" if pitch.get("window_end") is None else f"{pitch['window_end']:.4f}",
                        "" if pitch.get("window_duration") is None else f"{pitch['window_duration']:.4f}",
                        "" if pitch.get("expected_note_hz") is None else f"{pitch['expected_note_hz']:.2f}",
                        "" if pitch.get("mean_abs_cents_from_note") is None else f"{pitch['mean_abs_cents_from_note']:.4f}",
                    ])
            
            self._log(f"Metadata exported to {json_path.name} and {csv_path.name}")
            
        except Exception as e:
            logger.error(f"Error exporting metadata: {e}", exc_info=True)
            self._log(f"Warning: Could not export metadata: {e}")


def main() -> None:
    root = tk.Tk()
    OptimizedAudioSplitter(root)
    root.mainloop()


if __name__ == "__main__":
    main()

