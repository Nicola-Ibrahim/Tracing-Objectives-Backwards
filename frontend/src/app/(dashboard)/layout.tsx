import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/layout/Sidebar";

export default function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <SidebarProvider>
            <div className="flex min-h-screen w-full">
                <AppSidebar />
                <main className="flex-1 overflow-auto bg-background transition-colors duration-500 p-10">
                    <div className="min-h-full rounded-[1rem] border border-border/50 bg-muted/10 backdrop-blur-sm p-3 md:p-5">
                        {children}
                    </div>
                </main>
            </div>
        </SidebarProvider>
    );
}
