# AGENTS.md - QTS-ARCHITECT Constitution

Este archivo define las reglas estrictas, estándares de codificación y arquitectura para el proyecto **QTS-ARCHITECT**.
**ATENCIÓN AGENTE:** Debes leer y obedecer estas reglas antes de escribir una sola línea de código. Ignorar estas directrices resultará en código rechazado.

## 1. Identidad del Proyecto & Misión
QTS-Architect es un sistema de trading cuantitativo institucional, modular y asíncrono.
* **Objetivo:** Alta performance, seguridad de tipos y diseño desacoplado.
* **Mentalidad:** Piensa como un arquitecto de software senior en un fondo de cobertura (Hedge Fund). Prioriza la robustez sobre la velocidad de desarrollo.

## 2. Stack Tecnológico Obligatorio (HARD CONSTRAINTS)
No sustituyas estas tecnologías a menos que se te indique explícitamente.

| Dominio | Tecnología | Regla Específica |
| :--- | :--- | :--- |
| **Configuración** | `Hydra` + `OmegaConf` | Todo debe ser inyectable vía `_target_`. No hardcodees constantes. |
| **DataFrames** | `Polars` | **PROHIBIDO PANDAS**. Usa `LazyFrame` por defecto para optimización. |
| **Validación** | `Pydantic v2` | Modelos inmutables (`frozen=True`). |
| **Ejecución** | `asyncio` + `CCXT` | Todo I/O debe ser no bloqueante (async/await). |
| **Logging** | `structlog` | Salida JSON estructurada. No uses `print()` ni `logging` estándar. |
| **Testing** | `pytest` | Usa `pytest-asyncio` para corrutinas. Mocks estrictos para Gateways. |

## 3. Principios de Arquitectura

### 3.1 Protocolos sobre ABCs
* **Regla:** Prefiere `typing.Protocol` para definir interfaces en lugar de `abc.ABC`.
* **Razón:** Queremos *Structural Subtyping* (Duck Typing estático) para máxima flexibilidad y desacoplamiento.

### 3.2 Inmutabilidad por Defecto
* Todos los objetos de dominio (MarketData, Signal, Order) deben ser modelos Pydantic con `model_config = ConfigDict(frozen=True)`.
* Evita el estado mutable compartido. Usa mensajería o patrones de estado explícitos.

### 3.3 Tipado Estricto (Strict Typing)
* **No `Any` implícito.** El código debe pasar `mypy --strict`.
* Define tipos explícitos para argumentos y retornos.
* Usa `Optional[T]` en lugar de asumir `None`.

### 3.4 Configuración Externalizada (Hydra)
* Si creas una nueva clase de servicio, debe diseñarse para ser instanciada por Hydra.
* Ejemplo correcto: `def __init__(self, db_url: str):`
* Ejemplo incorrecto: `def __init__(self): self.db_url = os.getenv(...)`

## 4. Guía de Implementación de Código

### Estructura de Directorios
Respeta la siguiente jerarquía:
```text
qts_architect/
├── conf/                 # Configuración Hydra (YAMLs)
├── src/
│   ├── core/             # Interfaces (Protocols), Domain Models
│   ├── data/             # Loaders, Transforms (Polars)
│   ├── agents/           # Strategies, Supervisor, Risk logic
│   ├── execution/        # OMS, EMS, Gateways (CCXT)
│   └── utils/            # Logging, Helpers
├── tests/                # Unitarios e Integración
└── notebooks/            # Research (Jupyter)
```
