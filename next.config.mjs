/** @type {import('next').NextConfig} */
const nextConfig = {
    outputFileTracingIncludes: {
        "/api/predict": ["./model/**/*"],
        "/perbandingan": ["./model/**/*"],
    },
};
export default nextConfig;