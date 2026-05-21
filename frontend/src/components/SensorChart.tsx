import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, ReferenceLine, Brush
} from 'recharts';

interface SensorChartProps {
  data: { cycle: number; rul: number | null }[];
}

export const SensorChart: React.FC<SensorChartProps> = ({ data }) => {
  return (
    <div className="bg-panel rounded-2xl border border-panel-border shadow-lg p-6 h-96">
      <h3 className="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-widest flex items-center">
        <span className="w-2 h-2 rounded-full bg-accent mr-2 animate-pulse"></span>
        Sensors Trend (Predicted RUL vs Time)
      </h3>
      
      <div className="w-full h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" vertical={false} />
            <XAxis 
              dataKey="cycle" 
              stroke="#4B5563" 
              tick={{ fill: '#9CA3AF', fontSize: 12, fontFamily: 'monospace' }} 
              tickLine={false}
              domain={['dataMin', 'dataMax']}
            />
            <YAxis 
              stroke="#4B5563" 
              tick={{ fill: '#9CA3AF', fontSize: 12, fontFamily: 'monospace' }} 
              tickLine={false}
              domain={[0, 200]}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#111827', borderColor: '#1F2937', borderRadius: '8px' }}
              itemStyle={{ color: '#E5E7EB', fontFamily: 'monospace' }}
              labelStyle={{ color: '#9CA3AF', marginBottom: '8px' }}
              formatter={(value) => [`${(value as number).toFixed(1)} cycles`, 'Predicted RUL']}
              labelFormatter={(label) => `Cycle ${label}`}
            />
            
            <ReferenceLine y={50} stroke="#10b981" strokeDasharray="3 3" opacity={0.5} label={{ position: 'insideTopLeft', value: 'Healthy Bound', fill: '#10b981', fontSize: 10 }} />
            <ReferenceLine y={15} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} label={{ position: 'insideBottomLeft', value: 'Critical Bound', fill: '#ef4444', fontSize: 10 }} />

            <Line 
              type="monotone" 
              dataKey="rul" 
              stroke="#06b6d4" 
              strokeWidth={3}
              dot={{ r: 3, fill: '#111827', stroke: '#06b6d4', strokeWidth: 2 }}
              activeDot={{ r: 6, fill: '#06b6d4' }}
              isAnimationActive={false} 
            />
            
            <Brush 
              dataKey="cycle" 
              height={30} 
              stroke="#4B5563" 
              fill="#111827"
              startIndex={Math.max(0, data.length - 30)}
              tickFormatter={(val) => `Cy ${val}`}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
