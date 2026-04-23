import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/reset': 'http://localhost:7860',
      '/step': 'http://localhost:7860',
      '/health': 'http://localhost:7860',
      '/tasks': 'http://localhost:7860'
    }
  }
})
