/// <reference types="vite/client" />

declare module '*.css' {
  const content: { [className: string]: string }
  export default content
}

interface ImportMetaEnv {
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

interface Window {
  __INFORMITY_DESKTOP__?: boolean
  __INFORMITY_API_BASE__?: string
  __INFORMITY_API_TOKEN__?: string
}
