#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de seguridad y parada de emergencia
"""

import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

class EstadoSistema(Enum):
    """Máquina de estados del sistema"""
    INICIALIZANDO = auto()
    CALIBRANDO = auto()
    WARMUP = auto()
    CONTROL_ACTIVO = auto()
    DEGRADADO = auto()      # Operación con sensores limitados
    EMERGENCIA = auto()     # Parada activa
    ERROR_CRITICO = auto()  # Fallo irrecuperable

class SafetySystem:
    """
    Sistema de seguridad con watchdog y parada de emergencia
    """
    
    def __init__(self):
        self.estado = EstadoSistema.INICIALIZANDO
        self._lock = threading.RLock()
        self._emergency_stop = threading.Event()
        self._watchdog_last_ping = time.time()
        self._watchdog_timeout = 1.0  # segundos
        
        # Callbacks de emergencia
        self._emergency_callbacks: list[Callable] = []
        
        # Thread de watchdog
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._running = True
        
    def start(self):
        """Inicia el sistema de seguridad"""
        self._watchdog_thread.start()
        print("[SAFETY] Sistema de seguridad activado")
        
    def _watchdog_loop(self):
        """Monitorea la salud del sistema principal"""
        while self._running:
            time.sleep(0.1)
            
            with self._lock:
                elapsed = time.time() - self._watchdog_last_ping
                
            if elapsed > self._watchdog_timeout:
                if self.estado not in [EstadoSistema.EMERGENCIA, EstadoSistema.ERROR_CRITICO]:
                    self.trigger_emergency(f"Watchdog timeout: {elapsed:.1f}s sin respuesta")
                    
    def ping(self):
        """Señal de vida del sistema principal"""
        with self._lock:
            self._watchdog_last_ping = time.time()
            
    def register_emergency_callback(self, callback: Callable):
        """Registra función a llamar en emergencia (ej: parar motores)"""
        self._emergency_callbacks.append(callback)
        
    def trigger_emergency(self, razon: str):
        """Activa parada de emergencia"""
        with self._lock:
            if self.estado == EstadoSistema.EMERGENCIA:
                return  # Ya en emergencia
                
            self.estado = EstadoSistema.EMERGENCIA
            self._emergency_stop.set()
            
        print(f"\n{'='*60}")
        print(f"PARADA DE EMERGENCIA ACTIVADA")
        print(f"Razón: {razon}")
        print(f"{'='*60}\n")
        
        # Ejecutar callbacks
        for cb in self._emergency_callbacks:
            try:
                cb()
            except Exception as e:
                print(f"[SAFETY] Error en callback de emergencia: {e}")
                
    def set_estado(self, nuevo_estado: EstadoSistema):
        """Cambia el estado del sistema con validación"""
        with self._lock:
            # Validar transiciones permitidas
            transiciones_validas = {
                EstadoSistema.INICIALIZANDO: [EstadoSistema.CALIBRANDO, EstadoSistema.ERROR_CRITICO],
                EstadoSistema.CALIBRANDO: [EstadoSistema.WARMUP, EstadoSistema.ERROR_CRITICO],
                EstadoSistema.WARMUP: [EstadoSistema.CONTROL_ACTIVO, EstadoSistema.ERROR_CRITICO],
                EstadoSistema.CONTROL_ACTIVO: [EstadoSistema.DEGRADADO, EstadoSistema.EMERGENCIA],
                EstadoSistema.DEGRADADO: [EstadoSistema.CONTROL_ACTIVO, EstadoSistema.EMERGENCIA],
                EstadoSistema.EMERGENCIA: [EstadoSistema.INICIALIZANDO],  # Solo reinicio manual
            }
            
            permitidas = transiciones_validas.get(self.estado, [])
            if nuevo_estado not in permitidas and nuevo_estado != self.estado:
                print(f"[SAFETY] Transición no permitida: {self.estado.name} -> {nuevo_estado.name}")
                return False
                
            viejo = self.estado
            self.estado = nuevo_estado
            print(f"[SAFETY] Estado: {viejo.name} -> {nuevo_estado.name}")
            return True
            
    def get_estado(self) -> EstadoSistema:
        with self._lock:
            return self.estado
            
    def is_emergency(self) -> bool:
        return self._emergency_stop.is_set()
        
    def reset_emergency(self):
        """Solo para reinicio manual supervisado"""
        with self._lock:
            if self.estado == EstadoSistema.EMERGENCIA:
                self._emergency_stop.clear()
                self.estado = EstadoSistema.INICIALIZANDO
                self._watchdog_last_ping = time.time()
                print("[SAFETY] Emergencia reseteada - requiere reinicio completo")
                
    def shutdown(self):
        """Apagado ordenado"""
        self._running = False
        self._watchdog_thread.join(timeout=2.0)