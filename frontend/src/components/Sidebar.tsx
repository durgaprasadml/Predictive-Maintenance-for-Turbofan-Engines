import React from 'react';
import { Activity, Power, AlertTriangle, CheckCircle } from 'lucide-react';
import type { EngineData } from '../hooks/useTurbofanStream';

interface SidebarProps {
  enginesMap: Record<number, EngineData>;
  selectedEngine: number | null;
  onSelectEngine: (id: number) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ enginesMap, selectedEngine, onSelectEngine }) => {
  const engineIds = Object.keys(enginesMap).map(Number).sort((a, b) => a - b);

  return (
    <aside className="w-72 bg-panel border-r border-panel-border flex flex-col h-screen overflow-hidden">
      <div className="p-6 border-b border-panel-border flex items-center gap-3">
        <Activity className="text-accent h-6 w-6" />
        <h1 className="text-lg font-bold text-white tracking-widest uppercase">Turbofan<br/><span className="text-accent text-sm">Telemetry</span></h1>
      </div>
      
      <div className="p-4 flex-1 overflow-y-auto">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-4">Active Streams</h2>
        
        {engineIds.length === 0 ? (
          <div className="text-center p-4 text-gray-500 text-sm border border-dashed border-gray-700 rounded-lg">
            No engines connected.<br />Waiting for stream...
          </div>
        ) : (
          <div className="space-y-2">
            {engineIds.map(id => {
              const engine = enginesMap[id];
              const isSelected = selectedEngine === id;
              
              let statusColor = "text-gray-400";
              let StatusIcon = Power;
              
              if (engine.status === 'Healthy') {
                statusColor = "text-healthy";
                StatusIcon = CheckCircle;
              } else if (engine.status === 'Warning') {
                statusColor = "text-warning";
                StatusIcon = AlertTriangle;
              } else if (engine.status === 'Critical') {
                statusColor = "text-critical";
                StatusIcon = AlertTriangle;
              } else if (engine.status === 'Buffering') {
                statusColor = "text-accent";
                StatusIcon = Activity;
              }

              return (
                <button
                  key={id}
                  onClick={() => onSelectEngine(id)}
                  className={`w-full flex items-center justify-between p-3 rounded-xl transition-all duration-200 border ${
                    isSelected 
                      ? 'bg-gray-800/50 border-gray-600 shadow-md' 
                      : 'bg-transparent border-transparent hover:bg-gray-800/30'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <StatusIcon className={`h-4 w-4 ${statusColor} ${engine.status === 'Buffering' ? 'animate-pulse' : ''}`} />
                    <span className="font-mono text-sm font-semibold text-gray-200">ENG-{id.toString().padStart(3, '0')}</span>
                  </div>
                  
                  {engine.status === 'Buffering' ? (
                    <span className="text-xs font-mono text-gray-500">{engine.buffer_size}/30</span>
                  ) : (
                    <span className={`text-xs font-mono font-bold ${statusColor}`}>
                      {Math.round(engine.current_rul || 0)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>
      
      <div className="p-4 border-t border-panel-border text-xs text-gray-500 text-center font-mono">
        SYSTEM READY
      </div>
    </aside>
  );
};
