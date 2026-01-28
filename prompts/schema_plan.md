{
    "type": "OBJECT",
    "properties": {
        "chapters": {
            "type": "ARRAY",
            "description": "Список всех глав романа.",
            "items": {
                "type": "OBJECT",
                "description": "Представляет одну главу.",
                "properties": {
                    "number": {
                        "type": "STRING",
                        "description": "Номер главы в виде строки (например, '1', '2')."
                    },
                    "title": {
                        "type": "STRING",
                        "description": "Название главы (например, 'Глава 1')."
                    },
                    "scenes": {
                        "type": "ARRAY",
                        "description": "Список подробных описаний сцен для этой главы.",
                        "items": {
                            "type": "STRING",
                            "description": "Подробное описание одной сцены (100-120 слов), включая дату, время, действие, цель и связь с сюжетом."
                        }
                    }
                },
                "required": [
                    "number",
                    "title",
                    "scenes"
                ]
            }
        }
    },
    "required": [
        "chapters"
    ]
}
