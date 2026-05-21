import React from 'react';
import { motion } from 'framer-motion';

interface FailureAlertProps {
  isVisible: boolean;
  rul: number | null;
}

export const FailureAlert: React.FC<FailureAlertProps> = ({ isVisible, rul }) => {
  if (!isVisible) return null;

  return (
    <motion.div 
      className="fixed inset-0 pointer-events-none z-50 flex items-center justify-center p-4 bg-critical/10"
      initial={{ opacity: 0 }}
      animate={{ opacity: [0, 1, 0] }}
      transition={{ 
        duration: 1.5, 
        repeat: Infinity, 
        ease: "easeInOut" 
      }}
    >
      <div className="bg-panel border-2 border-critical shadow-2xl rounded-2xl p-8 max-w-md w-full pointer-events-auto">
        <div className="flex flex-col items-center text-center">
          <motion.div 
            className="w-20 h-20 bg-critical/20 rounded-full flex items-center justify-center mb-6"
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <div className="w-12 h-12 bg-critical rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-2xl">!</span>
            </div>
          </motion.div>
          
          <h2 className="text-2xl font-bold text-white mb-2 uppercase tracking-widest">
            Critical Alert
          </h2>
          <p className="text-gray-300 font-mono mb-6">
            Predicted Failure Soon.<br />
            Recommended immediate maintenance.
            <br/><br/>
            <span className="text-critical font-bold text-xl">{Math.round(rul || 0)} Cycles Remaining</span>
          </p>
        </div>
      </div>
    </motion.div>
  );
};
