"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navigation() {
    const pathname = usePathname();

    const links = [
        { href: "/data", label: "Data Hub" },
        { href: "/generate", label: "Candidate Generator" },
    ];

    return (
        <nav className="mb-10 flex space-x-4">
            {links.map((link) => {
                const isActive = pathname === link.href;
                return (
                    <Link
                        key={link.href}
                        href={link.href}
                        className={`px-4 py-2 rounded-lg transition-all duration-200 ${isActive
                            ? "bg-primary text-white shadow-md shadow-primary/20"
                            : "text-slate-600 hover:bg-slate-100 hover:text-primary"
                            }`}
                    >
                        {link.label}
                    </Link>
                );
            })}
        </nav>
    );
}
