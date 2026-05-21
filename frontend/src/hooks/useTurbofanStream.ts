import { useState, useEffect, useRef, useCallback } from 'react';

// Common interfaces
export interface EnginePrediction {
  unit_number: number;
  predicted_rul: number;
  status: 'Healthy' | 'Warning' | 'Critical';
  confidence_band: { lower: number; upper: number };
  timestamp: string;
}

export interface BufferingStatus {
  unit_number: number;
  buffer_size: number;
  cycles_needed: number;
  time_cycles: number;
}

export interface EngineData {
  unit_number: number;
  current_rul: number | null;
  status: 'Healthy' | 'Warning' | 'Critical' | 'Buffering';
  buffer_size: number;
  history: { cycle: number; rul: number | null }[];
}

export function useTurbofanStream(engineIds: number[] | 'all') {
  const [isConnected, setIsConnected] = useState(false);
  const [enginesData, setEnginesData] = useState<Record<number, EngineData>>({});
  const wsRef = useRef<WebSocket | null>(null);

  const initWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    
    // Connect to viewer websocket using env var or fallback
    const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/live';
    const ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
      setIsConnected(true);
      // Subscribe to requested engines
      ws.send(JSON.stringify({ 
        action: engineIds === 'all' ? 'subscribe_all' : 'subscribe',
        engines: engineIds === 'all' ? [] : engineIds
      }));
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Auto reconnect
      setTimeout(initWebSocket, 2000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle heartbeat
        if (data.type === 'heartbeat') {
          ws.send(JSON.stringify({ action: 'ping' }));
          return;
        }

        if (data.type === 'buffering') {
          setEnginesData(prev => {
            const current = prev[data.unit_number] || { 
              unit_number: data.unit_number, current_rul: null, status: 'Buffering', history: [], buffer_size: 0 
            };
            return {
              ...prev,
              [data.unit_number]: {
                ...current,
                buffer_size: data.buffer_size,
                status: 'Buffering'
              }
            };
          });
        }

        if (data.type === 'prediction') {
          setEnginesData(prev => {
            const current = prev[data.unit_number] || { 
              unit_number: data.unit_number, current_rul: null, status: 'Buffering', history: [], buffer_size: 30 
            };
            
            // Keep last 500 data points for charting
            const newHistory = [...current.history, { 
              cycle: (current.history.length > 0 ? current.history[current.history.length - 1].cycle + 1 : 30),
              rul: data.predicted_rul 
            }].slice(-500);

            return {
              ...prev,
              [data.unit_number]: {
                ...current,
                current_rul: data.predicted_rul,
                status: data.status,
                buffer_size: 30, // By definition, prediction means buffer full
                history: newHistory
              }
            };
          });
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    wsRef.current = ws;
  }, [engineIds]);

  useEffect(() => {
    initWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [initWebSocket]);

  return { isConnected, enginesData };
}
