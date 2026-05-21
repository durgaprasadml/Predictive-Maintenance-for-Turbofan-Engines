import React from 'react';
import { motion } from 'framer-motion';

interface RULGaugeProps {
  rul: number | null;
  bufferSize: number;
  maxRul?: number;
}

export const RULGauge: React.FC<RULGaugeProps> = ({ rul, bufferSize, maxRul = 200 }) => {
  const isBuffering = rul === null || bufferSize < 30;
  // Cap percentage at 1
  const percentage = isBuffering ? (bufferSize / 30) : Math.min(Math.max(rul / maxRul, 0), 1);
  
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - percentage * circumference;

  let colorClass = "text-accent";
  if (!isBuffering) {
    if (rul >= 50) colorClass = "text-healthy";
    else if (rul >= 15) colorClass = "text-warning";
    else colorClass = "text-critical";
  }

  return (
    <div className="relative flex flex-col items-center justify-center p-6 bg-panel rounded-2xl border border-panel-border shadow-lg">
      <h3 className="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-widest">
        {isBuffering ? "Buffering Data" : "Remaining Useful Life"}
      </h3>
      
      <div className="relative flex items-center justify-center w-48 h-48">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="96"
            cy="96"
            r={radius}
            stroke="currentColor"
            strokeWidth="12"
            fill="transparent"
            className="text-gray-800"
          />
          <motion.circle
            cx="96"
            cy="96"
            r={radius}
            stroke="currentColor"
            strokeWidth="12"
            fill="transparent"
            className={colorClass}
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            strokeLinecap="round"
          />
        </svg>
        
        <div className="absolute flex flex-col items-center justify-center">
          {isBuffering ? (
            <>
              <span className="text-3xl font-mono text-gray-300 font-bold">{bufferSize}/30</span>
              <span className="text-xs text-gray-500 uppercase mt-1">Cycles</span>
            </>
          ) : (
            <>
              <span className="text-5xl font-mono text-white font-bold">{Math.round(rul)}</span>
              <span className="text-xs text-gray-500 uppercase mt-1">Cycles Left</span>
            </>
          )}
        </div>
      </div>
      
      {!isBuffering && (
        <div className="mt-6 flex gap-2">
          <span className={`px-3 py-1 rounded border text-xs font-mono font-bold uppercase tracking-wide
            ${rul >= 50 ? 'border-healthy text-healthy bg-healthy/10' : 
              rul >= 15 ? 'border-warning text-warning bg-warning/10' : 
              'border-critical text-critical bg-critical/10'}`}>
            {rul >= 50 ? 'Healthy' : rul >= 15 ? 'Warning' : 'Critical'}
          </span>
        </div>
      )}
    </div>
  );
};
