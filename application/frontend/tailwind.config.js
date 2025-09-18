/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      keyframes: {
        pulse1: {
          '0%, 100%': { transform: 'scale(1) translate(-50%, -50%)' },
          '50%': { transform: 'scale(1.1) translate(-45%, -45%)' },
        },
        ping1: {
          '75%, 100%': {
            transform: 'scale(1.5)',
            opacity: '0',
          },
        },
        pulse2: {
          '0%, 100%': { transform: 'scale(1.1) translate(-45%, -45%)' },
          '50%': { transform: 'scale(1) translate(-50%, -50%)' },
        },
        ping2: {
          '25%, 50%': {
            transform: 'scale(1.5)',
            opacity: '0',
          },
          '0%, 75%, 100%': {
            transform: 'scale(1)',
            opacity: '0.75',
          },
        },
      },
      animation: {
        'pulse1': 'pulse1 2s ease-in-out infinite',
        'pulse2': 'pulse2 2s ease-in-out infinite',
        'ping1': 'ping1 2s ease-in-out infinite',
        'ping2': 'ping2 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}