#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor ToF dual con fusión robusta y detección de fallas
"""

import time
import threading
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple

try:
    import VL53L1X
    TOF_AVAILABLE = True
except ImportError:
    TOF_AVAILABLE = False
    print("[ToF] VL53L1X no disponible - modo simulación")

class DualToFSensor:
    """
    Gestión de dos sensores VL53L1X con:
    - Fusión ponderada por confianza
    - Detección de outliers estadístico
    - Calibración de offset por sensor
    - Fallback gracefully
    """
    
    def __init__(self, bus1: int = 1, bus2: int = 0, 
                 addr: int = 0x29, target_mm: float = 150.0):
        self.bus1 = bus1
        self.bus2 = bus2
        self.addr = addr
        self.target = target_mm
        
        # Sensores
        self.sensor1 = None
        self.sensor2 = None
        
        # Buffers de filtrado (median filter para robustez)
        self.buf1 = deque(maxlen=10)
        self.buf2 = deque(maxlen=10)
        
        # Estadísticas para detección de outliers
        self.stats1 = {'mean': target_mm, 'std': 10.0, 'n': 0}
        self.stats2 = {'mean': target_mm, 'std': 10.0, 'n': 0}
        
        # Offsets de calibración
        self.offset1 = 0.0
        self.offset2 = 0.0
        
        # Threading
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Estado actual
        self._state = {
            'z1_raw': None,
            'z1_filtered': None,
            'z1_confianza': 0.0,
            'z2_raw': None,
            'z2_filtered': None,
            'z2_confianza': 0.0,
            'z_fusionado': None,
            'z_fusionado_confianza': 0.0,
            'modo_fusion': 'none',  # single1, single2, average, weighted, safe_min
            'timestamp': 0.0
        }
        
        self._inicializar_sensores()
        
    def _inicializar_sensores(self):
        """Inicializa ambos sensores ToF"""
        if not TOF_AVAILABLE:
            print("[ToF] Modo simulación - no se inicializan sensores físicos")
            return
            
        # Sensor 1
        try:
            self.sensor1 = VL53L1X.VL53L1X(i2c_bus=self.bus1, i2c_address=self.addr)
            self.sensor1.open()
            self.sensor1.start_ranging(1)  # Modo corto/medio
            print(f"[ToF] ✓ Sensor 1 inicializado (bus {self.bus1})")
        except Exception as e:
            print(f"[ToF] ✗ Sensor 1 falló: {e}")
            self.sensor1 = None
            
        # Sensor 2
        try:
            self.sensor2 = VL53L1X.VL53L1X(i2c_bus=self.bus2, i2c_address=self.addr)
            self.sensor2.open()
            self.sensor2.start_ranging(1)
            print(f"[ToF] ✓ Sensor 2 inicializado (bus {self.bus2})")
        except Exception as e:
            print(f"[ToF] ✗ Sensor 2 falló: {e}")
            self.sensor2 = None
            
    def _leer_sensor(self, sensor, buf, stats, offset_mm: float) -> Tuple[Optional[float], float]:
        """
        Lee un sensor con filtrado y detección de outliers
        
        Returns:
            (valor_filtrado, confianza)
        """
        if sensor is None:
            return None, 0.0
            
        try:
            distancia = sensor.get_distance()
            if distancia is None or distancia <= 0 or distancia > 4000:
                return None, 0.0
                
            # Aplicar offset de calibración
            distancia = float(distancia) + offset_mm
            
            # Detección de outlier (descartar si está muy lejos de la media histórica)
            if stats['n'] > 10:
                z_score = abs(distancia - stats['mean']) / (stats['std'] + 1e-6)
                if z_score > 3.0:  # Más de 3 desviaciones estándar
                    # Aceptar pero con baja confianza
                    confianza = 0.3
                else:
                    confianza = 1.0 - (z_score / 3.0) * 0.5  # 0.5 a 1.0
            else:
                confianza = 0.8  # Confianza media hasta tener estadísticas
                
            buf.append(distancia)
            
            # Filtro de mediana (más robusto que media ante outliers)
            if len(buf) >= 3:
                valor_filtrado = float(np.median(buf))
            else:
                valor_filtrado = distancia
                
            # Actualizar estadísticas (usar valor filtrado para estabilidad)
            stats['mean'] = 0.9 * stats['mean'] + 0.1 * valor_filtrado
            if len(buf) > 1:
                stats['std'] = 0.9 * stats['std'] + 0.1 * np.std(list(buf)[-5:])
            stats['n'] += 1
            
            return valor_filtrado, confianza
            
        except Exception as e:
            return None, 0.0
            
    def start(self):
        """Inicia thread de adquisición en background"""
        if self._thread is not None and self._thread.is_alive():
            return
            
        if self.sensor1 is None and self.sensor2 is None:
            print("[ToF] No hay sensores disponibles - no se inicia thread")
            return
            
        self._stop.clear()
        self._thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self._thread.start()
        print("[ToF] Thread de adquisición iniciado")
        
    def _acquisition_loop(self):
        """Loop continuo de adquisición"""
        periodo = 1.0 / 20.0  # 20 Hz
        
        while not self._stop.is_set():
            inicio = time.monotonic()
            
            # Leer ambos sensores
            z1, conf1 = self._leer_sensor(self.sensor1, self.buf1, self.stats1, self.offset1)
            z2, conf2 = self._leer_sensor(self.sensor2, self.buf2, self.stats2, self.offset2)
            
            # Fusión inteligente
            z_fusion, conf_fusion, modo = self._fusionar(z1, conf1, z2, conf2)
            
            with self._lock:
                self._state.update({
                    'z1_raw': z1,
                    'z1_filtered': z1,
                    'z1_confianza': conf1,
                    'z2_raw': z2,
                    'z2_filtered': z2,
                    'z2_confianza': conf2,
                    'z_fusionado': z_fusion,
                    'z_fusionado_confianza': conf_fusion,
                    'modo_fusion': modo,
                    'timestamp': time.time()
                })
                
            # Esperar siguiente ciclo
            transcurrido = time.monotonic() - inicio
            sleep_time = max(0, periodo - transcurrido)
            time.sleep(sleep_time)
            
    def _fusionar(self, z1, c1, z2, c2) -> Tuple[Optional[float], float, str]:
        """
        Algoritmo de fusión con múltiples estrategias según disponibilidad
        
        Prioridad:
        1. Ambos disponibles y consistentes -> promedio ponderado
        2. Discrepancia grande -> modo seguro (mínimo = más cercano)
        3. Solo uno disponible -> usar ese con confianza reducida
        4. Ninguno -> None
        """
        disponibles = [(z, c, i) for i, (z, c) in enumerate([(z1, c1), (z2, c2)], 1) 
                      if z is not None and c > 0.3]
                      
        if len(disponibles) == 0:
            return None, 0.0, 'none'
            
        if len(disponibles) == 1:
            z, c, idx = disponibles[0]
            return z, c * 0.7, f'single{idx}'  # Penalizar por ser solo uno
            
        # Ambos disponibles
        (z_a, c_a, idx_a), (z_b, c_b, idx_b) = disponibles
        
        # Verificar consistencia
        diff = abs(z_a - z_b)
        
        if diff < 20.0:  # Consistentes (< 2cm)
            # Promedio ponderado por confianza
            peso_total = c_a + c_b
            z_fus = (z_a * c_a + z_b * c_b) / peso_total
            c_fus = min(1.0, peso_total / 2.0)  # Máx 1.0
            return z_fus, c_fus, 'weighted'
            
        elif diff < 50.0:  # Discrepancia moderada
            # Usar el de mayor confianza
            if c_a > c_b:
                return z_a, c_a * 0.8, f'single{idx_a}_preferred'
            else:
                return z_b, c_b * 0.8, f'single{idx_b}_preferred'
                
        else:  # Discrepancia grande (> 5cm) - modo seguro
            # En impresión 3D, el sensor que mide MENOR distancia está más cerca
            # de la verdad (el otro podría estar viendo algo detrás)
            z_safe = min(z_a, z_b)
            # Confianza baja porque no sabemos cuál está bien
            return z_safe, 0.4, 'safe_min'
            
    def get_state(self) -> Dict:
        """Retorna estado actual thread-safe"""
        with self._lock:
            return dict(self._state)
            
    def calibrar_offset(self, distancia_real_mm: float, muestras: int = 50):
        """
        Calibración: colocar objeto a distancia_real_mm y ejecutar
        """
        print(f"[ToF] Calibrando contra referencia de {distancia_real_mm}mm...")
        
        offsets1 = []
        offsets2 = []
        
        for _ in range(muestras):
            if self.sensor1:
                try:
                    d = self.sensor1.get_distance()
                    if d:
                        offsets1.append(distancia_real_mm - d)
                except:
                    pass
                    
            if self.sensor2:
                try:
                    d = self.sensor2.get_distance()
                    if d:
                        offsets2.append(distancia_real_mm - d)
                except:
                    pass
                    
            time.sleep(0.02)
            
        if offsets1:
            self.offset1 = np.median(offsets1)
            print(f"[ToF] Offset sensor 1: {self.offset1:+.2f}mm")
        if offsets2:
            self.offset2 = np.median(offsets2)
            print(f"[ToF] Offset sensor 2: {self.offset2:+.2f}mm")
            
    def stop(self):
        """Detiene adquisición y libera recursos"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            
        for sensor in [self.sensor1, self.sensor2]:
            if sensor:
                try:
                    sensor.stop_ranging()
                    sensor.close()
                except:
                    pass