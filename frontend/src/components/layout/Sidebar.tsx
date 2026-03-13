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
    SidebarFooter,
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
import { ThemeToggle } from "@/components/ui/ThemeToggle";

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
        <Sidebar collapsible="icon" {...props} className="border-r border-border bg-sidebar transition-colors duration-300">
            <SidebarHeader className="border-b border-border py-4 px-3">
                <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-2 font-bold text-foreground overflow-hidden font-heading">
                        <BrainCircuit className="h-6 w-6 text-indigo-500 shrink-0" />
                        <span className="truncate group-data-[collapsible=icon]:hidden">Tracing Objectives</span>
                    </div>
                    <Link
                        href="/"
                        className="group-data-[collapsible=icon]:hidden flex items-center gap-2 px-2 py-1.5 text-xs font-bold text-indigo-500 hover:bg-muted rounded-[1rem] transition-colors border border-indigo-500/20 font-heading"
                    >
                        <LayoutDashboard className="h-3.5 w-3.5" />
                        Back to Landing
                    </Link>
                </div>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel className="text-muted-foreground font-bold italic uppercase text-[10px] tracking-widest font-heading">Core Workspace</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Data Hub" isActive={pathname?.startsWith("/datasets")}>
                                    <Link href="/datasets" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/datasets") ? "text-indigo-500 font-bold font-heading" : "text-muted-foreground hover:text-foreground")}>
                                        <Database className={cn("h-4 w-4", pathname?.startsWith("/datasets") && "text-indigo-500")} />
                                        <span>Data Hub</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Transformation" isActive={pathname?.startsWith("/modeling/transformation-preview")}>
                                    <Link href="/modeling/transformation-preview" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/modeling/transformation-preview") ? "text-indigo-500 font-bold font-heading" : "text-muted-foreground hover:text-foreground")}>
                                        <Layers className={cn("h-4 w-4", pathname?.startsWith("/modeling/transformation-preview") && "text-indigo-500")} />
                                        <span>Transformation</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

                <SidebarGroup>
                    <SidebarGroupLabel className="text-muted-foreground font-bold italic uppercase text-[10px] tracking-widest font-heading">Inference Engine</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Train Engine" isActive={pathname?.startsWith("/inverse/train")}>
                                    <Link href="/inverse/train" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/inverse/train") ? "text-indigo-500 font-bold font-heading" : "text-muted-foreground hover:text-foreground")}>
                                        <BrainCircuit className={cn("h-4 w-4", pathname?.startsWith("/inverse/train") && "text-indigo-500")} />
                                        <span>Train Engine</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Inference Hub" isActive={pathname?.startsWith("/engines")}>
                                    <Link href="/engines" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/engines") ? "text-indigo-500 font-bold font-heading" : "text-muted-foreground hover:text-foreground")}>
                                        <Cpu className={cn("h-4 w-4", pathname?.startsWith("/engines") && "text-indigo-500")} />
                                        <span>Inference Hub</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Generate Candidates" isActive={pathname?.startsWith("/inverse/generate")}>
                                    <Link href="/inverse/generate" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/inverse/generate") ? "text-indigo-500 font-bold font-heading" : "text-muted-foreground hover:text-foreground")}>
                                        <Trello className={cn("h-4 w-4", pathname?.startsWith("/inverse/generate") && "text-indigo-500")} />
                                        <span>Generate Candidates</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

                <SidebarGroup>
                    <SidebarGroupLabel className="text-muted-foreground font-bold italic uppercase text-[10px] tracking-widest font-heading">Analytics</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild tooltip="Evaluation" isActive={pathname?.startsWith("/evaluation")}>
                                    <Link href="/evaluation" className={cn("transition-colors duration-200 flex items-center gap-2", pathname?.startsWith("/evaluation") ? "text-indigo-500 font-bold font-heading" : "text-muted-foreground hover:text-foreground")}>
                                        <LineChartIcon className={cn("h-4 w-4", pathname?.startsWith("/evaluation") && "text-indigo-500")} />
                                        <span>Performance Evaluation</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
            <SidebarFooter className="p-4 border-t border-border flex flex-row items-center justify-between group-data-[collapsible=icon]:justify-center">
                <div className="flex flex-col group-data-[collapsible=icon]:hidden">
                    <span className="text-[10px] font-bold text-foreground uppercase tracking-tighter font-heading">Theme Selection</span>
                    <span className="text-[8px] text-muted-foreground font-semibold uppercase tracking-widest">Interface Skin</span>
                </div>
                <ThemeToggle />
            </SidebarFooter>
            <SidebarRail />
        </Sidebar>
    );
}
