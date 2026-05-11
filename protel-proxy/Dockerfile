FROM node:18-alpine
WORKDIR /app
COPY package.json ./
RUN npm install
COPY server.js ./
EXPOSE 11435
CMD ["npm", "start"]
