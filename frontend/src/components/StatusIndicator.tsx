import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, ShieldCheck, Shield } from 'lucide-react';

interface StatusIndicatorProps {
  status: 'Healthy' | 'Warning' | 'Critical' | 'Buffering';
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status }) => {
  let bgColor = "bg-panel";
  let textColor = "text-gray-400";
  let borderColor = "border-panel-border";
  let Icon = Shield;
  let text = "INITIALIZING...";

  if (status === 'Healthy') {
    bgColor = "bg-healthy/10";
    textColor = "text-healthy";
    borderColor = "border-healthy/30";
    Icon = ShieldCheck;
    text = "SYSTEM NOMINAL";
  } else if (status === 'Warning') {
    bgColor = "bg-warning/10";
    textColor = "text-warning";
    borderColor = "border-warning/30";
    Icon = ShieldAlert;
    text = "MAINTENANCE WARNING";
  } else if (status === 'Critical') {
    bgColor = "bg-critical/20";
    textColor = "text-critical";
    borderColor = "border-critical/50";
    Icon = ShieldAlert;
    text = "CRITICAL FAILURE IMMINENT";
  } else if (status === 'Buffering') {
    bgColor = "bg-accent/10";
    textColor = "text-accent";
    borderColor = "border-accent/30";
    Icon = Shield;
    text = "BUFFERING SENSORS";
  }

  return (
    <div className={`p-6 rounded-2xl border ${borderColor} ${bgColor} transition-colors duration-500 shadow-lg flex items-center justify-between`}>
      <div>
        <h3 className="text-sm font-semibold text-gray-400 mb-1 uppercase tracking-widest">
          Engine Status
        </h3>
        <p className={`text-xl font-bold tracking-wider ${textColor}`}>
          {text}
        </p>
      </div>
      
      <div className="relative">
        <Icon className={`w-12 h-12 ${textColor}`} />
        {status === 'Critical' && (
          <motion.div
            className="absolute inset-0 bg-critical rounded-full"
            initial={{ scale: 1, opacity: 0.5 }}
            animate={{ scale: 2, opacity: 0 }}
            transition={{ duration: 1, repeat: Infinity }}
          />
        )}
        {status === 'Warning' && (
          <motion.div
            className="absolute inset-0 bg-warning rounded-full"
            initial={{ scale: 1, opacity: 0.4 }}
            animate={{ scale: 1.5, opacity: 0 }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        )}
      </div>
    </div>
  );
};
