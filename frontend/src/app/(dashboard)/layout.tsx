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
                <main className="flex-1 overflow-auto bg-background transition-colors duration-500 p-6">
                    <div className="min-h-full rounded-[2.5rem] border border-border/50 bg-muted/10 backdrop-blur-sm p-4 md:p-8">
                        {children}
                    </div>
                </main>
            </div>
        </SidebarProvider>
    );
}
