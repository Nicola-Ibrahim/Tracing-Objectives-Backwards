"use client";

import * as React from "react";
import {
    Database,
    LineChart as LineChartIcon,
    LayoutDashboard,
    BrainCircuit,
    Settings2,
    Trello,
    Cpu,
    Layers,
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
        title: "Inference Hub",
        url: "/engines",
        icon: Cpu,
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
                <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-2 font-semibold text-slate-900 overflow-hidden">
                        <BrainCircuit className="h-6 w-6 text-indigo-600 shrink-0" />
                        <span className="truncate group-data-[collapsible=icon]:hidden">Tracing Objectives</span>
                    </div>
                    <Link
                        href="/"
                        className="group-data-[collapsible=icon]:hidden flex items-center gap-2 px-2 py-1.5 text-xs font-bold text-indigo-600 hover:bg-slate-50 rounded-lg transition-colors border border-indigo-100"
                    >
                        <LayoutDashboard className="h-3.5 w-3.5" />
                        Back to Landing
                    </Link>
                </div>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel className="text-slate-500 font-medium italic uppercase text-[10px] tracking-widest">Core Workspace</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Data Hub" isActive={pathname?.startsWith("/datasets")}>
                                    <Link href="/datasets" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/datasets") ? "text-indigo-600" : "text-slate-600 hover:text-slate-900")}>
                                        <Database className={cn("h-4 w-4", pathname?.startsWith("/datasets") && "text-indigo-600")} />
                                        <span className="font-bold">Data Hub</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Transformation" isActive={pathname?.startsWith("/modeling/transformation-preview")}>
                                    <Link href="/modeling/transformation-preview" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/modeling/transformation-preview") ? "text-indigo-600" : "text-slate-600 hover:text-slate-900")}>
                                        <Layers className={cn("h-4 w-4", pathname?.startsWith("/modeling/transformation-preview") && "text-indigo-600")} />
                                        <span className="font-bold">Transformation</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

                <SidebarGroup>
                    <SidebarGroupLabel className="text-slate-500 font-medium italic uppercase text-[10px] tracking-widest">Inference Engine</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Train Engine" isActive={pathname?.startsWith("/inverse/train")}>
                                    <Link href="/inverse/train" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/inverse/train") ? "text-indigo-600" : "text-slate-600 hover:text-slate-900")}>
                                        <BrainCircuit className={cn("h-4 w-4", pathname?.startsWith("/inverse/train") && "text-indigo-600")} />
                                        <span className="font-bold">Train Engine</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Inference Hub" isActive={pathname?.startsWith("/engines")}>
                                    <Link href="/engines" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/engines") ? "text-indigo-600" : "text-slate-600 hover:text-slate-900")}>
                                        <Cpu className={cn("h-4 w-4", pathname?.startsWith("/engines") && "text-indigo-600")} />
                                        <span className="font-bold">Inference Hub</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Generate Candidates" isActive={pathname?.startsWith("/inverse/generate")}>
                                    <Link href="/inverse/generate" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/inverse/generate") ? "text-indigo-600" : "text-slate-600 hover:text-slate-900")}>
                                        <Trello className={cn("h-4 w-4", pathname?.startsWith("/inverse/generate") && "text-indigo-600")} />
                                        <span className="font-bold">Generate Candidates</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

                <SidebarGroup>
                    <SidebarGroupLabel className="text-slate-500 font-medium italic uppercase text-[10px] tracking-widest">Analytics</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Evaluation" isActive={pathname?.startsWith("/evaluation")}>
                                    <Link href="/evaluation" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/evaluation") ? "text-indigo-600" : "text-slate-600 hover:text-slate-900")}>
                                        <LineChartIcon className={cn("h-4 w-4", pathname?.startsWith("/evaluation") && "text-indigo-600")} />
                                        <span className="font-bold">Performance Evaluation</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
            <SidebarRail />
        </Sidebar>
    );
}
