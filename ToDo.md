# To do list to improve AWS deployment

## ~~Keep the README updated~~

- Should have build instructions
- Should have test instructions
- Should have deploy instructions
- Description of project
- Architecture of system

## Containerisation and orcarisation 

- Resolve dependancy issues regarding application (mostly tensorflow)

### Progress steps:

- install docker with command `sudo apt-get install docker.io`
- create docker file `nano Dockerfile`
- build image `docker build -t stockmodeldocker .`
- run image with `docker run stockmodeldocker`

## ~~SSH session management~~
- Run as a service

## Domain name

## Implement reverse proxy

## SSL/HTTPS

## Which ports should be used
- partially depends on SSL/HTTPS

## Logging/Logger
- How to know when something has crashed or gone wrong

## Uploaded files
- Currently deleted
- What happens if two users upload files at same time (race condition)

## Rate limiting

## Monitoring

## Sign in/Sign up/Login

## API
- Expose for others to build on