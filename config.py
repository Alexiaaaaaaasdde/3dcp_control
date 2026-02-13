#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración centralizada del sistema 3DCP
Todas las constantes modificables en un solo lugar
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List

# ==================== CONFIGURACIÓN HARDWARE ====================

@dataclass
class HardwareConfig:
    """Configuración de hardware físico"""
    # Cámaras
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 576
    CAMERA_FPS: int = 15
    USE_COLOR: bool = True
    
    # ToF
    TOF1_BUS: int = 1
    TOF2_BUS: int = 0
    TOF_ADDR: int = 0x29
    TOF_TARGET_MM: float = 150.0
    TOF_FILTER_WINDOW: int = 5  # Aumentado para más estabilidad
    TOF_READ_HZ: float = 20.0   # Aumentado para mejor respuesta
    
    # Referencia física
    ALTURA_REF_MM: float = 130.0
    PX_TO_MM_REF: float = 0.075
    
    # Seguridad
    VELOCIDAD_MIN_PCT: float = 30.0   # Más conservador que 10%
    VELOCIDAD_MAX_PCT: float = 150.0  # Más conservador que 200%
    
    # Tiempos críticos (segundos)
    WARMUP_FRAMES: int = 120
    TIMEOUT_FRAME: float = 0.5
    TIMEOUT_CRITICO: float = 2.0
    RETARDO_TRANSPORTE: float = 0.3  # Segundos boquilla->sensor

# ==================== CONFIGURACIÓN CONTROL ====================

@dataclass
class ControlConfig:
    """Parámetros del controlador"""
    # Banda muerta
    TOLERANCIA_MM: float = 0.5
    
    # Suavizado EMA
    ALPHA_SMOOTH: float = 0.15  # Más suave que 0.20
    
    # PID (ajustar según respuesta del concreto)
    KP: float = 25.0      # (mm/min)/mm - ganancia proporcional
    KI: float = 2.0       # Integral - elimina error estacionario
    KD: float = 5.0       # Derivativo - anticipa cambios
    
    # Anti-windup
    INTEGRAL_MAX_MM: float = 50.0  # Máxima acumulación de error integral
    
    # Límites de cambio (rate limiter)
    MAX_DELTA_VELOCIDAD: float = 100.0  # mm/min por ciclo

# ==================== CONFIGURACIÓN VISIÓN ====================

@dataclass
class VisionConfig:
    """Parámetros de procesamiento de imagen"""
    # ROI
    ROI_ENABLE: bool = True
    ROI_WIDTH_PERCENT: float = 0.5   # Más amplio para mejor detección
    ROI_HEIGHT_PERCENT: float = 0.5
    
    # Procesamiento
    UMBRAL_BINARIO: int = 80
    MIN_AREA_CONTORNO: int = 150
    MORPH_KERNEL_SIZE: int = 5
    PROCESS_HALF_RES: bool = True
    
    # Filtros de forma (filamento = alargado)
    MIN_ASPECT_RATIO: float = 1.5
    MAX_ASPECT_RATIO: float = 25.0
    MIN_SOLIDITY: float = 0.4
    
    # Detección de fallos (Kazemian 2019)
    FAILURE_THRESHOLD: int = 20  # Frames sin detección = rotura

# ==================== CONFIGURACIÓN LOGGING ====================

@dataclass
class LoggingConfig:
    """Configuración de logging"""
    LOG_DIR: str = "logs"
    FLUSH_INTERVAL: int = 10
    LOG_LEVEL: str = "INFO"
    
    # Columnas del CSV
    CSV_COLUMNS: List[str] = field(default_factory=lambda: [
        "timestamp",
        "estado_sistema",
        "ancho_medido_mm",
        "ancho_objetivo_mm",
        "error_mm",
        "altura_tof_mm",
        "altura_fusionada_mm",
        "velocidad_cmd_mmmin",
        "feedrate_pct",
        "gcode",
        "kp_activo",
        "ki_activo",
        "kd_activo",
        "confianza_vision",
        "confianza_tof",
        "fps",
        "latencia_ms",
        "modo_operacion"  # normal, degradado, emergencia
    ])

# Instancias globales (importar de aquí)
HW = HardwareConfig()
CTRL = ControlConfig()
VISION = VisionConfig()
LOG = LoggingConfig()

# Crear directorio de logs
os.makedirs(LOG.LOG_DIR, exist_ok=True)
