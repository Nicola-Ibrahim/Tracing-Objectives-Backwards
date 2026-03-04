"use client";

import * as React from "react";
import {
    Database,
    LineChart as LineChartIcon,
    LayoutDashboard,
    BrainCircuit,
    Settings2,
    Trello,
} from "lucide-react";

import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarRail,
} from "@/components/ui/sidebar";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const items = [
    {
        title: "Data Hub",
        url: "/datasets",
        icon: Database,
    },
    {
        title: "Train Engine",
        url: "/inverse/train",
        icon: BrainCircuit,
    },
    {
        title: "Generate Candidates",
        url: "/inverse/generate",
        icon: Trello,
    },
    {
        title: "Evaluation",
        url: "/evaluation",
        icon: LineChartIcon,
    },
];

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
    const pathname = usePathname();

    return (
        <Sidebar collapsible="icon" {...props} className="border-r border-slate-200">
            <SidebarHeader className="border-b border-slate-200 py-4 px-3">
                <div className="flex items-center gap-2 font-semibold text-slate-900 overflow-hidden">
                    <BrainCircuit className="h-6 w-6 text-indigo-600 shrink-0" />
                    <span className="truncate group-data-[collapsible=icon]:hidden">Tracing Objectives</span>
                </div>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel className="text-slate-500 font-medium">Core Workspace</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {items.map((item) => {
                                const isActive = pathname.startsWith(item.url);
                                return (
                                    <SidebarMenuItem key={item.title}>
                                        <SidebarMenuButton asChild tooltip={item.title} isActive={isActive}>
                                            <Link
                                                href={item.url}
                                                className={cn(
                                                    "transition-colors duration-200 flex items-center gap-2",
                                                    isActive ? "text-indigo-600" : "text-slate-600 hover:text-slate-900"
                                                )}
                                            >
                                                <item.icon className={cn("h-4 w-4", isActive && "text-indigo-600")} />
                                                <span>{item.title}</span>
                                            </Link>
                                        </SidebarMenuButton>
                                    </SidebarMenuItem>
                                );
                            })}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
            <SidebarRail />
        </Sidebar>
    );
}
