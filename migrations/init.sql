-- ============================================================================
-- Palet8 Agents API - Safe Migration Script
-- Uses IF NOT EXISTS to avoid errors and is idempotent
-- ============================================================================

-- ============================================================================
-- PHASE 1: Core Tables (Session, Asset, Template, AgentLog, ModelConfig, ModelStats, Task)
-- ============================================================================

-- Session table (for Shopify auth)
CREATE TABLE IF NOT EXISTS "Session" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "isOnline" BOOLEAN NOT NULL DEFAULT false,
    "scope" TEXT,
    "expires" TIMESTAMP(3),
    "accessToken" TEXT NOT NULL,
    "userId" BIGINT,
    "firstName" TEXT,
    "lastName" TEXT,
    "email" TEXT,
    "accountOwner" BOOLEAN NOT NULL DEFAULT false,
    "locale" TEXT,
    "collaborator" BOOLEAN DEFAULT false,
    "emailVerified" BOOLEAN DEFAULT false,

    CONSTRAINT "Session_pkey" PRIMARY KEY ("id")
);

-- Asset table
CREATE TABLE IF NOT EXISTS "Asset" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "imageUrl" TEXT,
    "thumbnailUrl" TEXT,
    "prompt" TEXT NOT NULL,
    "style" TEXT NOT NULL DEFAULT 'realistic',
    "dimensions" TEXT NOT NULL DEFAULT '1024x1024',
    "model" TEXT NOT NULL DEFAULT 'gemini',
    "status" TEXT NOT NULL DEFAULT 'processing',
    "cost" INTEGER NOT NULL DEFAULT 10,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "deletedAt" TIMESTAMP(3),

    CONSTRAINT "Asset_pkey" PRIMARY KEY ("id")
);

-- Asset indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Asset_taskId_key') THEN
        CREATE UNIQUE INDEX "Asset_taskId_key" ON "Asset"("taskId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Asset_shop_idx') THEN
        CREATE INDEX "Asset_shop_idx" ON "Asset"("shop");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Asset_status_idx') THEN
        CREATE INDEX "Asset_status_idx" ON "Asset"("status");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Asset_createdAt_idx') THEN
        CREATE INDEX "Asset_createdAt_idx" ON "Asset"("createdAt");
    END IF;
END $$;

-- Template table
CREATE TABLE IF NOT EXISTS "Template" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "promptText" TEXT NOT NULL,
    "style" TEXT NOT NULL DEFAULT 'realistic',
    "tags" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "language" TEXT NOT NULL DEFAULT 'en',
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdBy" TEXT,
    "usageCount" INTEGER NOT NULL DEFAULT 0,
    "acceptRate" DOUBLE PRECISION,
    "avgScore" DOUBLE PRECISION,
    "lastUsed" TIMESTAMP(3),
    "source" TEXT NOT NULL DEFAULT 'manual',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Template_pkey" PRIMARY KEY ("id")
);

-- Template indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Template_name_key') THEN
        CREATE UNIQUE INDEX "Template_name_key" ON "Template"("name");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Template_category_idx') THEN
        CREATE INDEX "Template_category_idx" ON "Template"("category");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Template_isActive_idx') THEN
        CREATE INDEX "Template_isActive_idx" ON "Template"("isActive");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Template_usageCount_idx') THEN
        CREATE INDEX "Template_usageCount_idx" ON "Template"("usageCount");
    END IF;
END $$;

-- AgentLog table
CREATE TABLE IF NOT EXISTS "AgentLog" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "agentName" TEXT NOT NULL,
    "input" JSONB NOT NULL,
    "output" JSONB NOT NULL,
    "reasoning" TEXT,
    "executionTime" INTEGER NOT NULL,
    "status" TEXT NOT NULL,
    "routingMode" TEXT,
    "usedLlm" BOOLEAN NOT NULL DEFAULT false,
    "confidence" DOUBLE PRECISION,
    "fallbackUsed" BOOLEAN NOT NULL DEFAULT false,
    "creditsUsed" INTEGER NOT NULL DEFAULT 0,
    "llmTokens" INTEGER,
    "modelName" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AgentLog_pkey" PRIMARY KEY ("id")
);

-- AgentLog indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'AgentLog_taskId_idx') THEN
        CREATE INDEX "AgentLog_taskId_idx" ON "AgentLog"("taskId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'AgentLog_shop_idx') THEN
        CREATE INDEX "AgentLog_shop_idx" ON "AgentLog"("shop");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'AgentLog_agentName_idx') THEN
        CREATE INDEX "AgentLog_agentName_idx" ON "AgentLog"("agentName");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'AgentLog_routingMode_idx') THEN
        CREATE INDEX "AgentLog_routingMode_idx" ON "AgentLog"("routingMode");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'AgentLog_createdAt_idx') THEN
        CREATE INDEX "AgentLog_createdAt_idx" ON "AgentLog"("createdAt");
    END IF;
END $$;

-- ModelConfig table
CREATE TABLE IF NOT EXISTS "ModelConfig" (
    "id" TEXT NOT NULL,
    "modelName" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "costPerGen" INTEGER NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "capabilities" JSONB NOT NULL,
    "priority" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ModelConfig_pkey" PRIMARY KEY ("id")
);

-- ModelConfig indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ModelConfig_modelName_key') THEN
        CREATE UNIQUE INDEX "ModelConfig_modelName_key" ON "ModelConfig"("modelName");
    END IF;
END $$;

-- ModelStats table
CREATE TABLE IF NOT EXISTS "ModelStats" (
    "id" TEXT NOT NULL,
    "modelName" TEXT NOT NULL,
    "bucket" TEXT NOT NULL,
    "impressions" INTEGER NOT NULL DEFAULT 0,
    "rewardMean" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "rewardVar" DOUBLE PRECISION NOT NULL DEFAULT 0.1,
    "lastUpdated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ModelStats_pkey" PRIMARY KEY ("id")
);

-- ModelStats indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ModelStats_modelName_bucket_key') THEN
        CREATE UNIQUE INDEX "ModelStats_modelName_bucket_key" ON "ModelStats"("modelName", "bucket");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ModelStats_bucket_idx') THEN
        CREATE INDEX "ModelStats_bucket_idx" ON "ModelStats"("bucket");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ModelStats_modelName_idx') THEN
        CREATE INDEX "ModelStats_modelName_idx" ON "ModelStats"("modelName");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ModelStats_lastUpdated_idx') THEN
        CREATE INDEX "ModelStats_lastUpdated_idx" ON "ModelStats"("lastUpdated");
    END IF;
END $$;

-- Task table
CREATE TABLE IF NOT EXISTS "Task" (
    "id" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "originalPrompt" TEXT NOT NULL,
    "userRequest" JSONB,
    "stages" JSONB NOT NULL,
    "promptJourney" JSONB NOT NULL,
    "totalDuration" INTEGER NOT NULL,
    "creditsCost" INTEGER NOT NULL,
    "performanceBreakdown" JSONB NOT NULL,
    "evaluationResults" JSONB,
    "generatedImageUrl" TEXT,
    "mockupUrls" JSONB,
    "finalPrompt" TEXT,
    "status" TEXT NOT NULL,
    "errorMessage" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" TIMESTAMP(3),
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Task_pkey" PRIMARY KEY ("id")
);

-- Task indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Task_taskId_key') THEN
        CREATE UNIQUE INDEX "Task_taskId_key" ON "Task"("taskId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Task_taskId_idx') THEN
        CREATE INDEX "Task_taskId_idx" ON "Task"("taskId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Task_shop_idx') THEN
        CREATE INDEX "Task_shop_idx" ON "Task"("shop");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Task_status_idx') THEN
        CREATE INDEX "Task_status_idx" ON "Task"("status");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Task_createdAt_idx') THEN
        CREATE INDEX "Task_createdAt_idx" ON "Task"("createdAt");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Task_completedAt_idx') THEN
        CREATE INDEX "Task_completedAt_idx" ON "Task"("completedAt");
    END IF;
END $$;

-- ============================================================================
-- PHASE 1 (Palet8 Agent Restructure): New models for multi-agent framework
-- Job, Conversation, ChatMessage, Design
-- ============================================================================

-- Job table (main orchestration entity)
CREATE TABLE IF NOT EXISTS "Job" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'INIT',
    "requirements" JSONB,
    "plan" JSONB,
    "prompt" JSONB,
    "evaluation" JSONB,
    "safetyCheck" JSONB,
    "creditCost" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Job_pkey" PRIMARY KEY ("id")
);

-- Job indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Job_userId_idx') THEN
        CREATE INDEX "Job_userId_idx" ON "Job"("userId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Job_status_idx') THEN
        CREATE INDEX "Job_status_idx" ON "Job"("status");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Job_createdAt_idx') THEN
        CREATE INDEX "Job_createdAt_idx" ON "Job"("createdAt");
    END IF;
END $$;

-- Conversation table
CREATE TABLE IF NOT EXISTS "Conversation" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "jobId" TEXT,
    "status" TEXT NOT NULL DEFAULT 'active',
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Conversation_pkey" PRIMARY KEY ("id")
);

-- Conversation indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Conversation_userId_idx') THEN
        CREATE INDEX "Conversation_userId_idx" ON "Conversation"("userId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Conversation_jobId_idx') THEN
        CREATE INDEX "Conversation_jobId_idx" ON "Conversation"("jobId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Conversation_status_idx') THEN
        CREATE INDEX "Conversation_status_idx" ON "Conversation"("status");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Conversation_createdAt_idx') THEN
        CREATE INDEX "Conversation_createdAt_idx" ON "Conversation"("createdAt");
    END IF;
END $$;

-- Add foreign key for Conversation.jobId -> Job.id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'Conversation_jobId_fkey'
    ) THEN
        ALTER TABLE "Conversation" ADD CONSTRAINT "Conversation_jobId_fkey"
        FOREIGN KEY ("jobId") REFERENCES "Job"("id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

-- ChatMessage table
CREATE TABLE IF NOT EXISTS "ChatMessage" (
    "id" TEXT NOT NULL,
    "conversationId" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "content" TEXT,
    "toolCalls" JSONB,
    "toolResult" JSONB,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ChatMessage_pkey" PRIMARY KEY ("id")
);

-- ChatMessage indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ChatMessage_conversationId_idx') THEN
        CREATE INDEX "ChatMessage_conversationId_idx" ON "ChatMessage"("conversationId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ChatMessage_role_idx') THEN
        CREATE INDEX "ChatMessage_role_idx" ON "ChatMessage"("role");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ChatMessage_createdAt_idx') THEN
        CREATE INDEX "ChatMessage_createdAt_idx" ON "ChatMessage"("createdAt");
    END IF;
END $$;

-- Add foreign key for ChatMessage.conversationId -> Conversation.id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ChatMessage_conversationId_fkey'
    ) THEN
        ALTER TABLE "ChatMessage" ADD CONSTRAINT "ChatMessage_conversationId_fkey"
        FOREIGN KEY ("conversationId") REFERENCES "Conversation"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- Design table
CREATE TABLE IF NOT EXISTS "Design" (
    "id" TEXT NOT NULL,
    "jobId" TEXT NOT NULL,
    "assetId" TEXT,
    "prompt" TEXT NOT NULL,
    "negativePrompt" TEXT,
    "modelUsed" TEXT NOT NULL,
    "parameters" JSONB,
    "evaluationScore" DOUBLE PRECISION,
    "evaluationData" JSONB,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "imageUrl" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Design_pkey" PRIMARY KEY ("id")
);

-- Design indexes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Design_jobId_idx') THEN
        CREATE INDEX "Design_jobId_idx" ON "Design"("jobId");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Design_status_idx') THEN
        CREATE INDEX "Design_status_idx" ON "Design"("status");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'Design_createdAt_idx') THEN
        CREATE INDEX "Design_createdAt_idx" ON "Design"("createdAt");
    END IF;
END $$;

-- Add foreign key for Design.jobId -> Job.id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'Design_jobId_fkey'
    ) THEN
        ALTER TABLE "Design" ADD CONSTRAINT "Design_jobId_fkey"
        FOREIGN KEY ("jobId") REFERENCES "Job"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- ============================================================================
-- VERIFICATION: Check migration results
-- ============================================================================

DO $$
DECLARE
    job_count INT;
    conv_count INT;
    msg_count INT;
    design_count INT;
BEGIN
    SELECT COUNT(*) INTO job_count FROM "Job";
    SELECT COUNT(*) INTO conv_count FROM "Conversation";
    SELECT COUNT(*) INTO msg_count FROM "ChatMessage";
    SELECT COUNT(*) INTO design_count FROM "Design";

    RAISE NOTICE '=== Agents API Migration Complete ===';
    RAISE NOTICE 'Job: % records', job_count;
    RAISE NOTICE 'Conversation: % records', conv_count;
    RAISE NOTICE 'ChatMessage: % records', msg_count;
    RAISE NOTICE 'Design: % records', design_count;
    RAISE NOTICE '=====================================';
END $$;
