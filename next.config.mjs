/** @type {import('next').NextConfig} */
const nextConfig = {
    outputFileTracingIncludes: {
        "/api/predict": ["./model/**/*"],
        "/api/predict-all": ["./model/**/*"],
        "/perbandingan": ["./model/**/*"],
    },
};
export default nextConfig;