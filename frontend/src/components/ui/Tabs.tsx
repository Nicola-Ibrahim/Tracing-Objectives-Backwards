import React from 'react';

interface Tab {
    id: string;
    label: string;
}

interface TabsProps {
    tabs: Tab[];
    activeTab: string;
    onChange: (id: string) => void;
    className?: string;
}

const Tabs: React.FC<TabsProps> = ({ tabs, activeTab, onChange, className = '' }) => {
    return (
        <div className={`flex space-x-1 p-1 bg-slate-100/50 rounded-lg ${className}`}>
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    onClick={() => onChange(tab.id)}
                    className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${activeTab === tab.id
                            ? 'bg-white text-slate-800 shadow-sm'
                            : 'text-slate-500 hover:text-slate-700 hover:bg-white/50'
                        }`}
                >
                    {tab.label}
                </button>
            ))}
        </div>
    );
};

export default Tabs;
