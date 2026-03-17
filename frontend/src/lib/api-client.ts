import axios from "axios";

const isServer = typeof window === "undefined";
const apiVersion = process.env.NEXT_PUBLIC_API_VERSION || "v1";

const rawBaseURL = isServer 
  ? (process.env.INTERNAL_API_URL || "http://localhost")
  : (process.env.NEXT_PUBLIC_API_URL || "");

// Ensure we have a clean base without trailing slashes
const cleanBase = rawBaseURL.replace(/\/+$/, "");

// Construct the full versioned path
// If rawBaseURL is "http://localhost", we want "http://localhost/api/v1"
// If rawBaseURL is "/api", we want "/api/v1"
// If rawBaseURL is empty, we want "/api/v1"
let fullBaseURL: string;

if (cleanBase.includes("/api/")) {
    fullBaseURL = cleanBase;
} else if (cleanBase.endsWith("/api")) {
    fullBaseURL = `${cleanBase}/${apiVersion}`;
} else {
    // Covers empty string, "http://localhost", or custom domain roots
    fullBaseURL = `${cleanBase}/api/${apiVersion}`;
}

fullBaseURL = fullBaseURL.replace(/\/+$/, "");

export const apiClient = axios.create({
  baseURL: fullBaseURL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Response interceptor for generic error handling
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    let message = error.message || "An unexpected error occurred";
    
    if (error.response?.data?.detail) {
      const detail = error.response.data.detail;
      if (typeof detail === 'object' && detail !== null && 'message' in detail) {
        message = (detail as any).message;
      } else if (typeof detail === 'string') {
        message = detail;
      }
    }

    console.error("[API Error]:", message);
    return Promise.reject(new Error(message));
  }
);
