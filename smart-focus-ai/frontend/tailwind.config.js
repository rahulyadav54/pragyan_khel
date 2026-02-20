/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Poppins', 'Inter', 'system-ui', 'sans-serif'],
      },
      screens: {
        xs: '320px',
        md: '768px',
        lg: '1024px',
        '2xl': '1440px',
      },
      colors: {
        app: {
          bg: '#0f172a',
          card: 'rgba(255,255,255,0.05)',
        },
        accent: {
          cyan: '#22d3ee',
          blue: '#3b82f6',
        }
      },
      boxShadow: {
        premium: '0 20px 60px rgba(2, 6, 23, 0.45)',
        neon: '0 8px 28px rgba(34, 211, 238, 0.20)',
      },
      animation: {
        float: 'float 6s ease-in-out infinite',
        'aurora-shift': 'auroraShift 16s ease-in-out infinite alternate',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        auroraShift: {
          '0%': { transform: 'translate3d(-8%, -2%, 0) scale(1)' },
          '100%': { transform: 'translate3d(8%, 6%, 0) scale(1.08)' },
        },
      }
    },
  },
  plugins: [],
}
