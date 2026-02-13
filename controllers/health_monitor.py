#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoreo de salud de todos los componentes del sistema
"""

import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum

class ComponenteSalud(Enum):
    OK = "OK"
    ADVERTENCIA = "ADVERTENCIA"
    CRITICO = "CRITICO"
    FALLA = "FALLA"

@dataclass
class EstadoComponente:
    estado: ComponenteSalud
    mensaje: str
    timestamp: float
    datos: dict = None

class HealthMonitor:
    """
    Monitorea la salud de cámaras, ToF y sistema de control
    """
    
    def __init__(self, safety_system):
        self.safety = safety_system
        self._lock = threading.RLock()
        
        # Históricos para detección de drift
        self._tof_history = deque(maxlen=200)
        self._ancho_history = deque(maxlen=100)
        self._fps_history = deque(maxlen=50)
        
        # Estados actuales
        self.cam0 = EstadoComponente(ComponenteSalud.OK, "Inicializando", time.time())
        self.cam1 = EstadoComponente(ComponenteSalud.OK, "Inicializando", time.time())
        self.tof1 = EstadoComponente(ComponenteSalud.OK, "Inicializando", time.time())
        self.tof2 = EstadoComponente(ComponenteSalud.OK, "Inicializando", time.time())
        self.control = EstadoComponente(ComponenteSalud.OK, "Inicializando", time.time())
        
        # Umbrales
        self.UMBRAL_FPS_CRITICO = 10.0
        self.UMBRAL_FPS_ADVERTENCIA = 20.0
        self.UMBRAL_TOF_DISCREPANCIA = 30.0  # mm
        self.UMBRAL_TOF_DRIFT = 50.0  # mm de cambio en 10 segundos
        self.TIMEOUT_FRAME = 0.5  # segundos
        
    def update_camera(self, cam_id: int, frame_timestamp: float, 
                      ancho_detectado: Optional[float], fps_actual: float):
        """Actualiza estado de cámara"""
        now = time.time()
        age = now - frame_timestamp
        
        with self._lock:
            # Determinar estado
            if age > 2.0:
                estado = ComponenteSalud.FALLA
                mensaje = f"Sin frames por {age:.1f}s"
            elif age > 0.5:
                estado = ComponenteSalud.CRITICO
                mensaje = f"Latencia alta: {age*1000:.0f}ms"
            elif fps_actual < self.UMBRAL_FPS_CRITICO:
                estado = ComponenteSalud.CRITICO
                mensaje = f"FPS crítico: {fps_actual:.1f}"
            elif fps_actual < self.UMBRAL_FPS_ADVERTENCIA:
                estado = ComponenteSalud.ADVERTENCIA
                mensaje = f"FPS bajo: {fps_actual:.1f}"
            else:
                estado = ComponenteSalud.OK
                mensaje = f"OK - {fps_actual:.1f} FPS"
                
            comp = EstadoComponente(estado, mensaje, now, {
                'fps': fps_actual,
                'latencia_ms': age * 1000,
                'ancho': ancho_detectado
            })
            
            if cam_id == 0:
                self.cam0 = comp
            else:
                self.cam1 = comp
                
            self._fps_history.append(fps_actual)
            
            # Detectar tendencia de degradación
            if len(self._fps_history) == 50:
                fps_trend = np.polyfit(range(50), self._fps_history, 1)[0]
                if fps_trend < -0.5:  # Bajando más de 0.5 FPS por muestra
                    print(f"[HEALTH] Advertencia: Tendencia de degradación de FPS ({fps_trend:.2f})")
                    
    def update_tof(self, z1: Optional[float], z2: Optional[float], 
                   z_selected: Optional[float]):
        """Actualiza estado de sensores ToF"""
        now = time.time()
        
        with self._lock:
            # Estados individuales
            if z1 is None:
                self.tof1 = EstadoComponente(ComponenteSalud.FALLA, "Sin lectura", now)
            else:
                self.tof1 = EstadoComponente(ComponenteSalud.OK, f"{z1:.1f} mm", now, {'z': z1})
                
            if z2 is None:
                self.tof2 = EstadoComponente(ComponenteSalud.FALLA, "Sin lectura", now)
            else:
                self.tof2 = EstadoComponente(ComponenteSalud.OK, f"{z2:.1f} mm", now, {'z': z2})
                
            # Análisis de consistencia
            if z1 is not None and z2 is not None:
                diff = abs(z1 - z2)
                self._tof_history.append((now, z1, z2, diff))
                
                # Discrepancia inmediata
                if diff > self.UMBRAL_TOF_DISCREPANCIA:
                    print(f"[HEALTH] CRÍTICO: Discrepancia ToF = {diff:.1f}mm (>{self.UMBRAL_TOF_DISCREPANCIA})")
                    self.tof1 = EstadoComponente(ComponenteSalud.CRITICO, 
                        f"Discrepancia {diff:.1f}mm con ToF2", now, {'z': z1, 'diff': diff})
                    self.tof2 = EstadoComponente(ComponenteSalud.CRITICO,
                        f"Discrepancia {diff:.1f}mm con ToF1", now, {'z': z2, 'diff': diff})
                        
                # Detección de drift gradual (análisis de ventana)
                if len(self._tof_history) >= 100:
                    self._check_drift()
                    
    def _check_drift(self):
        """Detecta drift gradual en sensores ToF"""
        # Comparar primera mitad vs segunda mitad del histórico
        mitad = len(self._tof_history) // 2
        hist_list = list(self._tof_history)
        
        z1_primera = np.mean([x[1] for x in hist_list[:mitad] if x[1] is not None])
        z1_segunda = np.mean([x[1] for x in hist_list[mitad:] if x[1] is not None])
        z2_primera = np.mean([x[2] for x in hist_list[:mitad] if x[2] is not None])
        z2_segunda = np.mean([x[2] for x in hist_list[mitad:] if x[2] is not None])
        
        drift1 = abs(z1_segunda - z1_primera) if z1_primera and z1_segunda else 0
        drift2 = abs(z2_segunda - z2_primera) if z2_primera and z2_segunda else 0
        
        if drift1 > self.UMBRAL_TOF_DRIFT:
            print(f"[HEALTH] Drift detectado en ToF1: {drift1:.1f}mm en ~10s")
        if drift2 > self.UMBRAL_TOF_DRIFT:
            print(f"[HEALTH] Drift detectado en ToF2: {drift2:.1f}mm en ~10s")
            
    def update_control(self, error_mm: float, velocidad_cmd: float, 
                       saturado: bool, en_deadband: bool):
        """Monitorea salud del lazo de control"""
        now = time.time()
        
        with self._lock:
            self._ancho_history.append((now, error_mm))
            
            # Detectar oscilación (error cambiando de signo rápidamente)
            if len(self._ancho_history) >= 20:
                errores = [e for _, e in self._ancho_history[-20:]]
                sign_changes = sum(1 for i in range(1, len(errores)) 
                                  if errores[i-1] * errores[i] < 0)
                if sign_changes > 8:  # Más de 8 cambios de signo en 20 muestras = oscilación
                    estado = ComponenteSalud.ADVERTENCIA
                    mensaje = f"Posible oscilación ({sign_changes} cambios de signo)"
                elif saturado:
                    estado = ComponenteSalud.ADVERTENCIA
                    mensaje = "Actuador saturado"
                else:
                    estado = ComponenteSalud.OK
                    mensaje = f"Error: {error_mm:+.2f}mm"
                    
                self.control = EstadoComponente(estado, mensaje, now, {
                    'error': error_mm,
                    'velocidad': velocidad_cmd,
                    'saturado': saturado
                })
                
    def get_system_health(self) -> Dict[str, EstadoComponente]:
        """Retorna estado completo del sistema"""
        with self._lock:
            return {
                'cam0': self.cam0,
                'cam1': self.cam1,
                'tof1': self.tof1,
                'tof2': self.tof2,
                'control': self.control
            }
            
    def is_critical(self) -> bool:
        """True si hay alguna condición crítica"""
        health = self.get_system_health()
        return any(h.estado in [ComponenteSalud.CRITICO, ComponenteSalud.FALLA] 
                  for h in health.values())
                  
    def can_operate(self) -> Tuple[bool, str]:
        """Determina si se puede operar y en qué modo"""
        health = self.get_system_health()
        
        # Cámara principal debe funcionar
        if health['cam0'].estado == ComponenteSalud.FALLA:
            return False, "Cámara principal fuera de servicio"
            
        # Al menos un ToF debe funcionar
        tof_ok = health['tof1'].estado != ComponenteSalud.FALLA or \
                 health['tof2'].estado != ComponenteSalud.FALLA
        if not tof_ok:
            return False, "Ambos sensores ToF fuera de servicio"
            
        # Modo degradado si hay advertencias
        advertencias = sum(1 for h in health.values() 
                          if h.estado == ComponenteSalud.ADVERTENCIA)
        if advertencias > 0:
            return True, f"Modo degradado ({advertencias} advertencias)"
            
        return True, "Operación normal"