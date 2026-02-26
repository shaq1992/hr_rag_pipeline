#!/bin/bash

SOURCES_DIR="../sources"
API_URL="http://localhost:8000/ingest"

# Check if the directory exists
if [ ! -d "$SOURCES_DIR" ]; then
    echo "Error: Sources directory $SOURCES_DIR not found."
    exit 1
fi

# Count the total number of PDFs
TOTAL_FILES=$(ls -1q "$SOURCES_DIR"/*.pdf 2>/dev/null | wc -l)
if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "No PDF files found in $SOURCES_DIR."
    exit 0
fi

echo -e "\n============================================================"
echo -e " \033[1mStarting Batch Ingestion for $TOTAL_FILES documents\033[0m"
echo -e "============================================================"

CURRENT_FILE=1
GLOBAL_CHUNKS=0
GLOBAL_LLM_CALLS=0

for FILE_PATH in "$SOURCES_DIR"/*.pdf; do
    FILENAME=$(basename "$FILE_PATH")
    echo -e "\n\033[1m[$CURRENT_FILE/$TOTAL_FILES] Processing:\033[0m $FILENAME"
    
    # We use process substitution `< <(...)` to feed the curl stream into the while loop.
    # This prevents the loop from executing in a subshell, allowing us to 
    # update the GLOBAL_CHUNKS and GLOBAL_LLM_CALLS variables persistently.
    while IFS= read -r line; do
        
        # Native bash substring checks to avoid adding 'jq' as a strict dependency
        if [[ "$line" == *"\"status\": \"parsing\""* ]]; then
            printf "  \033[33m%s\033[0m\r" "-> Parsing document via Unstructured API..."
            
        elif [[ "$line" == *"\"status\": \"start_chunks\""* ]]; then
            TOTAL_CHUNKS=$(echo "$line" | grep -o '"total": [0-9]*' | awk '{print $2}')
            printf "\n  \033[36m%s\033[0m\n" "-> Parsed successfully. Preparing $TOTAL_CHUNKS chunks..."
            
        elif [[ "$line" == *"\"status\": \"chunk_progress\""* ]]; then
            CURRENT=$(echo "$line" | grep -o '"current": [0-9]*' | awk '{print $2}')
            printf "  \033[34m%s\033[0m\r" "-> Upserting chunk: $CURRENT / $TOTAL_CHUNKS"
            
        elif [[ "$line" == *"\"status\": \"success\""* ]]; then
            CHUNKS=$(echo "$line" | grep -o '"chunks_upserted": [0-9]*' | awk '{print $2}')
            LLM=$(echo "$line" | grep -o '"llm_calls": [0-9]*' | awk '{print $2}')
            
            GLOBAL_CHUNKS=$((GLOBAL_CHUNKS + CHUNKS))
            GLOBAL_LLM_CALLS=$((GLOBAL_LLM_CALLS + LLM))
            
            printf "\n  \033[32m%s\033[0m\n" "-> [SUCCESS] Upserted: $CHUNKS chunks | LLM Calls: $LLM"
            
        elif [[ "$line" == *"\"status\": \"error\""* ]]; then
            DETAIL=$(echo "$line" | grep -o '"detail": "[^"]*"' | cut -d'"' -f4)
            printf "\n  \033[31m%s\033[0m\n" "-> [ERROR] $DETAIL"
        fi
        
    done < <(curl -N -s -X POST "$API_URL" -H "Content-Type: application/json" -d "{\"file_path\": \"/app/sources/$FILENAME\"}")
    
    ((CURRENT_FILE++))
done

echo -e "\n============================================================"
echo -e " \033[1mINGESTION BATCH COMPLETE\033[0m"
echo -e "============================================================"
echo " Total Documents Processed : $TOTAL_FILES"
echo " Total Chunks Upserted     : $GLOBAL_CHUNKS"
echo " Total LLM Calls           : $GLOBAL_LLM_CALLS"
echo -e "============================================================\n"
