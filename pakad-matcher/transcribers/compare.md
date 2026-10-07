# Pitch sources compared (training/validation only)

| source | misread ↓ | phrase ↑ (method) | directions ↑ (method) | nyas F1 ↑ (method) |
| --- | --- | --- | --- | --- |
| melodia | 0.559 | 0.744 (val-tuned) | 0.628 (learned + notation) | 0.409 (learned) |
| basic_pitch | 0.661 | 0.685 (read-then-match) | 0.585 (learned) | 0.207 (learned) |
| basic_pitch+demucs | 0.699 | 0.617 (read-then-match) | 0.595 (learned + notation) | 0.373 (learned) |
| ymt3plus | 0.776 | 0.498 (val-tuned) | 0.396 (learned) | 0.100 (learned) |
| yourmt3 | 0.701 | 0.546 (read-then-match) | 0.455 (learned) | 0.255 (learned) |
| yourmt3+demucs | 0.674 | — | — | — |
