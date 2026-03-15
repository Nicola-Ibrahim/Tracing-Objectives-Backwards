# Component Documentation Template

## Overview
[One-sentence description of the component's purpose]

## Usage
```tsx
import { MyComponent } from './MyComponent';

const Example = () => <MyComponent prop1="value" />;
```

## Props
| Prop | Type | Description |
| :--- | :--- | :---------- |
| `prop1` | `string` | Purpose of prop1 |

## Internal Logic
- **State**: Describes what state it manages.
- **Effects**: Describes side effects (API calls, subscriptions).

---

# API Endpoint Documentation Template

## `METHOD /path/to/endpoint`
[Brief description of what the endpoint does]

### Request Parameters
- `param1` (type): Description.

### Response Body
```json
{
  "status": "success",
  "data": {}
}
```

### Errors
- `400 Bad Request`: When X happens.
- `404 Not Found`: When Y is missing.
