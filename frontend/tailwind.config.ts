import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			fontFamily: {
				'inter': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
				'comic': ['Bangers', 'Comic Sans MS', 'Impact', 'sans-serif'],
			},
			colors: {
				// Design system colors using HSL variables
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				surface: 'hsl(var(--surface))',
				'surface-secondary': 'hsl(var(--surface-secondary))',
				'surface-hover': 'hsl(var(--surface-hover))',
				
				// Chat interface
				'chat-background': 'hsl(var(--chat-background))',
				'user-bubble': 'hsl(var(--user-bubble))',
				'user-bubble-foreground': 'hsl(var(--user-bubble-foreground))',
				'ai-bubble': 'hsl(var(--ai-bubble))',
				'ai-bubble-foreground': 'hsl(var(--ai-bubble-foreground))',
				
				// Primary colors
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))',
					hover: 'hsl(var(--primary-hover))',
				},
				
				// Spotify branding
				spotify: {
					DEFAULT: 'hsl(var(--spotify))',
					hover: 'hsl(var(--spotify-hover))',
				},
				
				// Borders
				border: 'hsl(var(--border))',
				'border-subtle': 'hsl(var(--border-subtle))',
				
				// Text hierarchy
				'text-primary': 'hsl(var(--text-primary))',
				'text-secondary': 'hsl(var(--text-secondary))',
				'text-muted': 'hsl(var(--text-muted))',
				
				// Interactive states
				hover: 'hsl(var(--hover))',
				active: 'hsl(var(--active))',
			},
			borderRadius: {
				lg: 'var(--radius-lg)',
				md: 'var(--radius)',
				sm: 'calc(var(--radius) - 2px)',
				xs: 'calc(var(--radius) - 4px)',
				full: 'var(--radius-full)',
			},
			spacing: {
				'chat': 'var(--space-chat)',
				'bubble': 'var(--space-bubble)',
			},
			transitionTimingFunction: {
				'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
				'spring': 'cubic-bezier(0.34, 1.56, 0.64, 1)',
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out'
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
