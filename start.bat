Rem reset variables to null
set docker_img_exists=
set docker_container_running=
set docker_container_exited=

echo "Checking if image is already installed..."

for /f %%i in ('docker images -q gp-image:latest') do set docker_img_exists=%%i

if "%docker_img_exists%"=="" docker build -t gp-image:latest .

for /f %%j in ('docker ps -q -f "name=gp"') do set docker_container_running=%%j
for /f %%k in ('docker ps -aq -f "status=exited" -f "name=gp"') do set docker_container_exited=%%k

Rem check if already running
if "%docker_container_running%"=="" (
  Rem check if container exists but is stopped
  if not "%docker_container_exited%"=="" (
    Rem in whih case, remove it
    docker rm gp
  )
  Rem start new container
  echo "Starting container..."
  docker run -it -p 4200:4200 --name gp gp-image:latest
) else (
  echo "Container is already running. Go to http://localhost:4200 to start the experiment."
)