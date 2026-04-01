import axios from "axios";

const isServer = typeof window === "undefined";
const apiVersion = process.env.API_VERSION || "v1";

const rawBaseURL = process.env.BACKEND_URL;

// Ensure we have a clean base without trailing slashes
const cleanBase = rawBaseURL?.replace(/\/+$/, "") || "";

// Construct the full versioned path
// If rawBaseURL is undefined, we want "/v1"
// If rawBaseURL is "http://localhost", we want "http://localhost/v1"
const fullBaseURL = `${cleanBase}/${apiVersion}`.replace(/\/+$/, "");

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
        message = (detail as Record<string, string>).message;
      } else if (typeof detail === 'string') {
        message = detail;
      }
    }

    console.error("[API Error]:", message);
    return Promise.reject(new Error(message));
  }
);
