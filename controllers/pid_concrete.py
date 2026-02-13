#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlador PID específico para dinámica de concreto
Incluye: compensación de retardo, anti-windup, rate limiter
"""

import time
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict

class ConcretePIDController:
    """
    PID optimizado para control de ancho en impresión 3D de concreto
    
    Características:
    - Compensación de retardo de transporte (Smith Predictor simplificado)
    - Anti-windup integral con back-calculation
    - Rate limiter en salida (cambios bruscos dañan calidad)
    - Filtro derivativo de primer orden
    - Banda muerta configurable
    """
    
    def __init__(self, 
                 velocidad_base: float,
                 ancho_objetivo: float,
                 kp: float = 25.0,
                 ki: float = 2.0,
                 kd: float = 5.0,
                 tiempo_retardo: float = 0.3,
                 banda_muerta: float = 0.5,
                 alpha_ema: float = 0.15,
                 integral_max: float = 50.0,
                 max_delta_vel: float = 100.0):
        
        # Parámetros de control
        self.velocidad_base = float(velocidad_base)
        self.ancho_objetivo = float(ancho_objetivo)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau_retardo = tiempo_retardo
        self.banda_muerta = banda_muerta
        self.alpha_ema = alpha_ema
        self.integral_max = integral_max
        self.max_delta_vel = max_delta_vel
        
        # Límites de seguridad
        self.v_min = velocidad_base * 0.3  # 30% mínimo
        self.v_max = velocidad_base * 1.5  # 150% máximo
        
        # Estado del controlador
        self.error_integral = 0.0
        self.error_anterior = 0.0
        self.derivada_filtrada = 0.0
        self.ultima_velocidad = velocidad_base
        self.ultimo_tiempo = time.time()
        
        # Buffer para compensación de retardo (Smith Predictor)
        # Guarda: (timestamp, velocidad_aplicada, ancho_medido_predicho)
        self.historial_control = deque(maxlen=100)
        
        # Métricas de desempeño
        self.ciclos_control = 0
        self.tiempo_saturado = 0.0
        self.tiempo_deadband = 0.0
        
    def _compensar_retardo(self, ancho_medido: float, timestamp: float) -> float:
        """
        Compensación simplificada de retardo de transporte
        Usa el modelo: el ancho actual es resultado de la velocidad 
        aplicada hace ~tau_retardo segundos
        """
        if len(self.historial_control) < 10:
            return ancho_medido  # No hay suficiente historial
            
        # Buscar el comando de velocidad que se aplicó hace tau_retardo
        tiempo_objetivo = timestamp - self.tau_retardo
        
        # Interpolar en el historial
        historial = list(self.historial_control)
        velocidad_pasada = None
        
        for i, (t, v, _) in enumerate(historial):
            if t >= tiempo_objetivo:
                if i == 0:
                    velocidad_pasada = v
                else:
                    # Interpolación lineal
                    t0, v0, _ = historial[i-1]
                    if t > t0:
                        factor = (tiempo_objetivo - t0) / (t - t0)
                        velocidad_pasada = v0 + factor * (v - v0)
                break
                
        if velocidad_pasada is None:
            return ancho_medido
            
        # Estimar ancho que se habría medido sin retardo
        # Modelo simplificado: cambio de 10% en velocidad -> cambio de ~1mm en ancho
        sensibilidad = 0.1  # mm por % de cambio de velocidad
        delta_v_pct = (self.ultima_velocidad - velocidad_pasada) / self.velocidad_base
        correccion = delta_v_pct * 100 * sensibilidad
        
        ancho_compensado = ancho_medido - correccion
        return ancho_compensado
        
    def compute(self, ancho_medido: Optional[float], 
                timestamp: Optional[float] = None) -> Tuple[str, Dict]:
        """
        Calcula comando de velocidad con PID completo
        
        Returns:
            gcode_cmd: Comando G-code (M220 Sxxx)
            datos: Diccionario con información detallada del control
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Sin medición - mantener última velocidad con decay suave
        if ancho_medido is None:
            v_decay = self.ultima_velocidad * 0.95 + self.velocidad_base * 0.05
            v_decay = np.clip(v_decay, self.v_min, self.v_max)
            self.ultima_velocidad = v_decay
            self._actualizar_historial(timestamp, v_decay, None)
            
            return f"M220 S{int(v_decay/self.velocidad_base*100)}", {
                'error_mm': None,
                'error_compensado': None,
                'P': 0, 'I': 0, 'D': 0,
                'velocidad_cmd': v_decay,
                'feedrate_pct': v_decay/self.velocidad_base*100,
                'saturado': False,
                'deadband': False,
                'modo': 'HOLD (sin medición)'
            }
            
        # Compensación de retardo
        ancho_compensado = self._compensar_retardo(ancho_medido, timestamp)
        
        # Calcular error
        error = ancho_compensado - self.ancho_objetivo
        
        # Banda muerta
        if abs(error) < self.banda_muerta:
            error_efectivo = 0.0
            en_deadband = True
            self.tiempo_deadband += 0.02  # Asumiendo 50Hz
        else:
            error_efectivo = error
            en_deadband = False
            
        # Tiempo transcurrido
        dt = timestamp - self.ultimo_tiempo
        self.ultimo_tiempo = timestamp
        if dt <= 0 or dt > 1.0:  # Protección contra pausas o primer ciclo
            dt = 0.02  # Asumir 50Hz por defecto
            
        # Término proporcional
        P = self.kp * error_efectivo
        
        # Término integral con anti-windup (back-calculation)
        # Solo integrar si no estamos saturados
        self.error_integral += error_efectivo * dt
        
        # Anti-windup: limitar integral
        self.error_integral = np.clip(self.error_integral, -self.integral_max, self.integral_max)
        I = self.ki * self.error_integral
        
        # Término derivativo con filtro
        if dt > 0:
            derivada_cruda = (error_efectivo - self.error_anterior) / dt
        else:
            derivada_cruda = 0
            
        # Filtro de primer orden: tau/(tau+s) donde tau = 1/(2*pi*fc)
        # fc = 10Hz -> tau = 0.016
        alpha_d = 0.2  # Equivalente a ~8Hz de corte
        self.derivada_filtrada = alpha_d * derivada_cruda + (1 - alpha_d) * self.derivada_filtrada
        D = self.kd * self.derivada_filtrada
        
        self.error_anterior = error_efectivo
        
        # PID total
        pid_output = P + I + D
        
        # Velocidad calculada (en unidades del sistema)
        v_calculada = self.velocidad_base + pid_output
        
        # Rate limiter (limitar cambio respecto a última velocidad)
        delta_v = v_calculada - self.ultima_velocidad
        delta_v_limitado = np.clip(delta_v, -self.max_delta_vel, self.max_delta_vel)
        v_rate_limited = self.ultima_velocidad + delta_v_limitado
        
        # Saturación final
        v_final = np.clip(v_rate_limited, self.v_min, self.v_max)
        
        # Detectar saturación
        saturado = (v_final != v_calculada)
        if saturado:
            self.tiempo_saturado += dt
            # Anti-windup: reducir integral cuando saturamos
            self.error_integral *= 0.95
            
        # Suavizado EMA final
        v_suave = self.alpha_ema * v_final + (1 - self.alpha_ema) * self.ultima_velocidad
        
        # Actualizar estado
        self.ultima_velocidad = v_suave
        self._actualizar_historial(timestamp, v_suave, ancho_compensado)
        self.ciclos_control += 1
        
        # Generar G-code
        feedrate_pct = (v_suave / self.velocidad_base) * 100
        gcode = f"M220 S{int(round(feedrate_pct))}"
        
        # Preparar datos de retorno
        datos = {
            'error_mm': error,
            'error_compensado': error_efectivo,
            'ancho_compensado': ancho_compensado,
            'P': P,
            'I': I,
            'D': D,
            'velocidad_cmd': v_suave,
            'feedrate_pct': feedrate_pct,
            'gcode': gcode,
            'saturado': saturado,
            'deadband': en_deadband,
            'rate_limited': abs(delta_v) > self.max_delta_vel,
            'integral_acumulado': self.error_integral,
            'modo': 'PID-ACTIVO' if not en_deadband else 'DEADBAND'
        }
        
        return gcode, datos
        
    def _actualizar_historial(self, timestamp: float, velocidad: float, 
                              ancho_predicho: Optional[float]):
        """Guarda historial para compensación de retardo"""
        self.historial_control.append((timestamp, velocidad, ancho_predicho))
        
    def get_status(self) -> str:
        """Retorna string de estado para display"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║           CONTROLADOR PID - CONCRETO 3DCP                   ║
╠══════════════════════════════════════════════════════════════╣
║  Base: {self.velocidad_base:6.1f} mm/min  |  Objetivo: {self.ancho_objetivo:5.2f} mm          ║
║  Kp={self.kp:5.2f}  Ki={self.ki:5.2f}  Kd={self.kd:5.2f}  |  Retardo: {self.tau_retardo*1000:4.0f}ms    ║
╠══════════════════════════════════════════════════════════════╣
║  Último error: {self.error_anterior:+6.2f} mm  |  Integral: {self.error_integral:+6.2f}      ║
║  Ciclos: {self.ciclos_control:5d}  |  Saturado: {self.tiempo_saturado:5.1f}s                ║
╚══════════════════════════════════════════════════════════════╝
"""
        
    def reset(self):
        """Resetea el controlador (para nuevo trabajo)"""
        self.error_integral = 0.0
        self.error_anterior = 0.0
        self.derivada_filtrada = 0.0
        self.ultima_velocidad = self.velocidad_base
        self.historial_control.clear()
        self.ciclos_control = 0
        self.tiempo_saturado = 0.0
        self.tiempo_deadband = 0.0
        print("[PID] Controlador reseteado")