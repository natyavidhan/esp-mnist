#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void app_main(void)
{
	char buf[128];


	while (1) {
		if (fgets(buf, sizeof(buf), stdin)) {
			printf("recieved: %s", buf);
		}
		// printf("Hello World!\n");
		vTaskDelay(pdMS_TO_TICKS(10));
	}
}
