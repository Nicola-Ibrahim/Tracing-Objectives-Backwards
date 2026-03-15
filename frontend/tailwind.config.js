/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./frontend/**/*.{html,js}"],
    theme: {
        extend: {
            colors: {
                primary: "#6366f1",
                "primary-hover": "#4f46e5",
                "accent-vibrant": "#ec4899",
                "glass-border": "rgba(255, 255, 255, 0.1)",
                "card-bg": "rgba(15, 23, 42, 0.65)",
                "text-main": "#f8fafc",
                "text-muted": "#94a3b8",
            },
            backdropBlur: {
                xs: "2px",
            }
        },
    },
    plugins: [],
}
